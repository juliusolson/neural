package neural

import (
	"fmt"
	"math"
	"neural/utils"

	"gonum.org/v1/gonum/mat"
)

/*
Cache stores calculations for convenience
*/
type Cache map[string]*mat.Dense

/*
Net implements training and testing
*/
type Net struct {
	Lr      float64
	K       int
	D       int
	DHidden int
	W       *mat.Dense
	B1      *mat.Dense
	B2      *mat.Dense
	C       *mat.Dense
	cache   Cache
}

/*
InitParams initializes weights
*/
func (nn *Net) InitParams() {
	nn.W = utils.RandomMatrix(nn.DHidden, nn.D, math.Sqrt(2.0/float64(nn.D)))
	nn.C = utils.RandomMatrix(nn.K, nn.DHidden, math.Sqrt(2.0/float64(nn.DHidden)))
	nn.B1 = utils.RandomMatrix(nn.DHidden, 1, 1.0)
	nn.B2 = utils.RandomMatrix(nn.K, 1, 1.0)

	nn.cache = make(map[string]*mat.Dense)
}

/*
Forward implements forward propagation
*/
func (nn *Net) Forward(x *mat.Dense) *mat.Dense {
	var Z, H, U mat.Dense
	Z.Mul(nn.W, x)
	Z.Add(&Z, nn.B1)
	H.Apply(utils.Sigmoid, &Z)
	U.Mul(nn.C, &H)
	U.Add(&U, nn.B2)

	nn.cache["U"] = &U
	nn.cache["H"] = &H
	nn.cache["Z"] = &Z

	return utils.Softmax(&U)
}

/*
Backward calulates the gradients
*/
func (nn *Net) Backward(out, x *mat.Dense, y int) {
	var dPdU, dPdC, Delta, dPdB1, dPdW mat.Dense
	dPdU.CloneFrom(out)
	dPdU.Set(y, 0, dPdU.At(y, 0)-1)
	dPdC.Mul(&dPdU, nn.cache["H"].T())
	Delta.Mul(nn.C.T(), &dPdU)
	var tmp mat.Dense
	tmp.Apply(utils.BackSigmoid, nn.cache["Z"])
	dPdB1.MulElem(&Delta, &tmp)
	dPdW.Mul(&dPdB1, x.T())

	nn.cache["dPdW"] = &dPdW
	nn.cache["dPdB1"] = &dPdB1
	nn.cache["dPdB2"] = &dPdU
	nn.cache["dPdC"] = &dPdC
	nn.cache["Delta"] = &Delta
}

/*
UpdateParams takes a step in the oppsite direction of the gradients
*/
func (nn *Net) UpdateParams() {
	dPdW := nn.cache["dPdW"]
	dPdW.Scale(nn.Lr, dPdW)
	dPdC := nn.cache["dPdC"]
	dPdC.Scale(nn.Lr, dPdC)
	dPdB1 := nn.cache["dPdB1"]
	dPdB1.Scale(nn.Lr, dPdB1)
	dPdB2 := nn.cache["dPdB2"]
	dPdB2.Scale(nn.Lr, dPdB2)

	nn.W.Sub(nn.W, dPdW)
	nn.C.Sub(nn.C, dPdC)
	nn.B1.Sub(nn.B1, dPdB1)
	nn.B2.Sub(nn.B2, dPdB2)
}

/*
Train runs SGD
*/
func (nn *Net) Train(X, Y *mat.Dense, epochs int) {
	N, _ := X.Dims()
	indices := utils.RandomIndices(N)

	for e := 0; e < epochs; e++ {
		for n, i := range indices {
			if n%1000 == 0 {
				fmt.Printf("\rEpoch %v, Progress:  %v / %v", e+1, n, N)
			}
			x := mat.DenseCopyOf(X.RowView(i))
			y := int(Y.At(i, 0))

			out := nn.Forward(x)
			nn.Backward(out, x, y)
			nn.UpdateParams()
		}
		fmt.Printf("\n")
	}
}

/*
Test the New on samples
*/
func (nn *Net) Test(X, Y *mat.Dense) {
	N, _ := X.Dims()
	corr := 0
	corrMap := make(map[int]int)
	for i := 0; i < 10; i++ {
		corrMap[i] = 0
	}

	for n := 0; n < N; n++ {
		x := mat.DenseCopyOf(X.RowView(n))
		y := int(Y.At(n, 0))
		out := nn.Forward(x)
		pred := utils.Argmax(out.ColView(0))
		if pred == y {
			corr++
		}
	}

	fmt.Printf("Accuracy: %v", float64(corr)/float64(N))
}

/*
NewNN initializes new Neural net
*/
func NewNN(lr float64, K, d, dH int) *Net {
	nn := &Net{
		Lr:      lr,
		K:       K,
		D:       d,
		DHidden: dH,
	}
	nn.InitParams()
	return nn

}
