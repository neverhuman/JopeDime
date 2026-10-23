#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p target/go-contract
cp target/go-metadata/{model.go,model.wasm,go.mod,go.sum} target/go-contract/
cp deploy/ci/go/contract_test.go.txt target/go-contract/contract_test.go
cd target/go-contract
go mod verify
go vet -mod=readonly ./...
go test -mod=readonly -count=1 -covermode=atomic -coverprofile=coverage.out -v ./...
go tool cover -func=coverage.out
