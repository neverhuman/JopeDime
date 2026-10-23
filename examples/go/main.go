package main
import("context";"fmt";"os";model "jope.generated/model")
func check(e error){if e!=nil{panic(e)}}
func main(){ctx:=context.Background();b,e:=os.ReadFile("model/model.wasm");check(e);m,e:=model.New(ctx,b);check(e);defer m.Close(ctx);check(m.Fit(ctx,[][]float64{{0},{1},{2},{3}},[]float64{1,3,5,7}));p,e:=m.Predict(ctx,[][]float64{{0.5},{2.5}});check(e);for _,v:=range p{fmt.Printf("%.17g\n",v)};s,e:=m.Score(ctx,[][]float64{{0.5},{2.5}},[]float64{2,6});check(e);fmt.Printf("R-squared: %.17g\n",s)}
