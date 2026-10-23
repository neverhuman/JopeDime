// Package model provides isolated numerical models with a pure Go Wasm runtime.
// Build the accompanying Rust kernel to Wasm once; no CGO or host imports are used.
package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/csv"
	"errors"
	"fmt"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"math"
	"sync"
	"unicode/utf8"
)

const maxElements = 16_777_216

var boundHeaders = __BOUND_HEADERS__

// Engine selects the default runtime or the portable interpreter.
type Engine uint8

const (
	Auto Engine = iota
	Interpreter
)

// Model serializes calls and owns its runtime. Do not copy a Model or change
// input slices while a call uses them. Close releases all instance storage.
type Model struct {
	mu    sync.Mutex
	k     kernel
	width int
}
type kernel interface {
	call(context.Context, string, ...uint64) (uint64, error)
	read(uint32, uint32) ([]byte, bool)
	write(uint32, []byte) bool
	close(context.Context) error
}
type wasmKernel struct {
	r wazero.Runtime
	m api.Module
}

func (k *wasmKernel) call(ctx context.Context, name string, args ...uint64) (uint64, error) {
	v, e := k.m.ExportedFunction(name).Call(ctx, args...)
	if e != nil {
		return 0, e
	}
	return v[0], nil
}
func (k *wasmKernel) read(p, n uint32) ([]byte, bool) { return k.m.Memory().Read(p, n) }
func (k *wasmKernel) write(p uint32, b []byte) bool   { return k.m.Memory().Write(p, b) }
func (k *wasmKernel) close(ctx context.Context) error { return k.r.Close(ctx) }

// New creates an independent model from compiled Wasm bytes.
func New(ctx context.Context, b []byte) (*Model, error) { return NewWithEngine(ctx, b, Auto) }

// NewWithEngine validates the complete ABI, refuses imports, limits linear
// memory to 512 MiB and enables cancellation of running kernel calls.
func NewWithEngine(ctx context.Context, b []byte, engine Engine) (*Model, error) {
	cfg := wazero.NewRuntimeConfig()
	switch engine {
	case Auto:
	case Interpreter:
		cfg = wazero.NewRuntimeConfigInterpreter()
	default:
		return nil, errors.New("unknown engine")
	}
	r := wazero.NewRuntimeWithConfig(ctx, cfg.WithMemoryLimitPages(8192).WithCloseOnContextDone(true))
	fail := func(e error) (*Model, error) { _ = r.Close(context.Background()); return nil, e }
	c, e := r.CompileModule(ctx, b)
	if e != nil {
		return fail(e)
	}
	if len(c.ImportedFunctions()) != 0 || len(c.ImportedMemories()) != 0 {
		return fail(errors.New("host imports forbidden"))
	}
	if _, ok := c.ExportedMemories()["memory"]; !ok {
		return fail(errors.New("missing memory"))
	}
	arities := map[string]int{"model_abi_version": 0, "model_values": 1, "model_labels": 1, "model_text": 1, "model_output": 0, "model_output_len": 0, "model_fit": 2, "model_predict": 2, "model_predict_text": 0}
	for name, n := range arities {
		f, ok := c.ExportedFunctions()[name]
		if !ok || len(f.ParamTypes()) != n || !bytes.Equal(f.ResultTypes(), []api.ValueType{api.ValueTypeI32}) {
			return fail(fmt.Errorf("invalid ABI function %s", name))
		}
		for _, t := range f.ParamTypes() {
			if t != api.ValueTypeI32 {
				return fail(errors.New("invalid ABI parameter"))
			}
		}
	}
	m, e := r.InstantiateModule(ctx, c, wazero.NewModuleConfig().WithStartFunctions())
	if e != nil {
		return fail(e)
	}
	k := &wasmKernel{r, m}
	v, e := k.call(ctx, "model_abi_version")
	if e != nil {
		return fail(e)
	}
	if v != 1 {
		return fail(errors.New("unsupported ABI version"))
	}
	return &Model{k: k}, nil
}
func (m *Model) ready() error {
	if m.k == nil {
		return errors.New("model closed")
	}
	return nil
}
func (m *Model) call(ctx context.Context, name string, args ...uint64) (uint64, error) {
	v, e := m.k.call(ctx, name, args...)
	if e != nil {
		_ = m.k.close(context.Background())
		m.k = nil
		return 0, e
	}
	return v, nil
}
func shape(x [][]float64, w int) (int, error) {
	if len(x) > maxElements {
		return 0, errors.New("too many rows")
	}
	if w == 0 && len(x) > 0 {
		w = len(x[0])
	}
	if w <= 0 || w > maxElements || len(x) > maxElements/w {
		return 0, errors.New("invalid dimensions")
	}
	for _, r := range x {
		if len(r) != w {
			return 0, errors.New("ragged or mismatched rows")
		}
	}
	return w, nil
}
func labels(y []float64, n int) error {
	if len(y) != n {
		return errors.New("target count mismatch")
	}
	for _, v := range y {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("non-finite target")
		}
	}
	return nil
}
func (m *Model) transfer(ctx context.Context, name string, x [][]float64, n int) error {
	p, e := m.call(ctx, name, uint64(n))
	if e != nil {
		return e
	}
	if p == 0 || p > math.MaxUint32 {
		return errors.New("allocation refused")
	}
	for _, r := range x {
		b := make([]byte, len(r)*8)
		for i, v := range r {
			binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(v))
		}
		if !m.k.write(uint32(p), b) {
			return errors.New("input outside memory")
		}
		p += uint64(len(b))
	}
	return nil
}
func (m *Model) status(ctx context.Context, name string, args ...uint64) error {
	v, e := m.call(ctx, name, args...)
	if e != nil {
		return e
	}
	if v != 0 {
		return fmt.Errorf("%s refused input (%d)", name, v)
	}
	return nil
}
func (m *Model) output(ctx context.Context, n int) ([]float64, error) {
	v, e := m.call(ctx, "model_output_len")
	if e != nil {
		return nil, e
	}
	if v != uint64(n) {
		return nil, errors.New("output count mismatch")
	}
	p, e := m.call(ctx, "model_output")
	if e != nil {
		return nil, e
	}
	if p > math.MaxUint32 {
		return nil, errors.New("invalid output address")
	}
	b, ok := m.k.read(uint32(p), uint32(n*8))
	if !ok {
		return nil, errors.New("output outside memory")
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[i*8:]))
	}
	return out, nil
}

// Fit trains or refits a numeric recipe; fitted artifacts refuse training.
func (m *Model) Fit(ctx context.Context, x [][]float64, y []float64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e := m.ready(); e != nil {
		return e
	}
	if boundHeaders != nil {
		return errors.New("fitted artifact cannot train")
	}
	if len(x) == 0 {
		return errors.New("empty training rows")
	}
	w, e := shape(x, 0)
	if e != nil {
		return e
	}
	if e = labels(y, len(x)); e != nil {
		return e
	}
	if e = m.transfer(ctx, "model_values", x, len(x)*w); e != nil {
		return e
	}
	if e = m.transfer(ctx, "model_labels", [][]float64{y}, len(y)); e != nil {
		return e
	}
	if e = m.status(ctx, "model_fit", uint64(len(x)), uint64(w)); e != nil {
		return e
	}
	m.width = w
	return nil
}
func (m *Model) predict(ctx context.Context, x [][]float64) ([]float64, error) {
	if e := m.ready(); e != nil {
		return nil, e
	}
	if boundHeaders != nil || m.width == 0 {
		return nil, errors.New("fit numeric recipe first")
	}
	if _, e := shape(x, m.width); e != nil {
		return nil, e
	}
	if e := m.transfer(ctx, "model_values", x, len(x)*m.width); e != nil {
		return nil, e
	}
	if e := m.status(ctx, "model_predict", uint64(len(x)), uint64(m.width)); e != nil {
		return nil, e
	}
	return m.output(ctx, len(x))
}

// Predict returns copied predictions. Empty batches are valid after Fit.
func (m *Model) Predict(ctx context.Context, x [][]float64) ([]float64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.predict(ctx, x)
}

type csvBuffer struct{ data []byte }

func (b *csvBuffer) Write(p []byte) (int, error) {
	if len(p) > maxElements-len(b.data) {
		return 0, errors.New("CSV transfer limit")
	}
	b.data = append(b.data, p...)
	return len(p), nil
}

// PredictStrings applies the saved schema and codecs to UTF-8 string cells.
func (m *Model) PredictStrings(ctx context.Context, rows [][]string) ([]float64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e := m.ready(); e != nil {
		return nil, e
	}
	if boundHeaders == nil {
		return nil, errors.New("string input requires fitted artifact")
	}
	if len(rows) > maxElements {
		return nil, errors.New("too many rows")
	}
	var b csvBuffer
	w := csv.NewWriter(&b)
	if e := w.Write(boundHeaders); e != nil {
		return nil, e
	}
	for _, r := range rows {
		if len(r) != len(boundHeaders) {
			return nil, errors.New("string width mismatch")
		}
		for _, s := range r {
			if !utf8.ValidString(s) {
				return nil, errors.New("invalid UTF-8")
			}
			if len(s) > maxElements {
				return nil, errors.New("cell transfer limit")
			}
		}
		if e := w.Write(r); e != nil {
			return nil, e
		}
	}
	w.Flush()
	if e := w.Error(); e != nil {
		return nil, e
	}
	p, e := m.call(ctx, "model_text", uint64(len(b.data)))
	if e != nil {
		return nil, e
	}
	if p == 0 || p > math.MaxUint32 {
		return nil, errors.New("text allocation refused")
	}
	if !m.k.write(uint32(p), b.data) {
		return nil, errors.New("text outside memory")
	}
	if e = m.status(ctx, "model_predict_text"); e != nil {
		return nil, e
	}
	return m.output(ctx, len(rows))
}

// Score returns R-squared. Empty and constant targets score zero.
func (m *Model) Score(ctx context.Context, x [][]float64, y []float64) (float64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e := m.ready(); e != nil {
		return 0, e
	}
	if e := labels(y, len(x)); e != nil {
		return 0, e
	}
	p, e := m.predict(ctx, x)
	if e != nil {
		return 0, e
	}
	if len(y) == 0 {
		return 0, nil
	}
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(len(y))
	if math.IsInf(mean, 0) {
		return 0, errors.New("target mean overflow")
	}
	total, residual := 0.0, 0.0
	for i, v := range y {
		total += (v - mean) * (v - mean)
		residual += (v - p[i]) * (v - p[i])
	}
	if total == 0 {
		return 0, nil
	}
	s := 1 - residual/total
	if math.IsNaN(s) || math.IsInf(s, 0) {
		return 0, errors.New("non-finite score")
	}
	return s, nil
}

// Close is idempotent and can run concurrently with other model methods.
func (m *Model) Close(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.k == nil {
		return nil
	}
	k := m.k
	m.k = nil
	return k.close(ctx)
}
