package utils

import (
	"encoding/csv"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

/*
	Read data
	Activation funcs
	Etc.
*/

func Sigmoid(_, _ int, v float64) float64 {
	return math.Exp(v) / (1 + math.Exp(v))
}

func BackSigmoid(_, _ int, v float64) float64 {
	s := Sigmoid(0, 0, v)
	return s * (1 - s)
}

func Softmax(m *mat.Dense) *mat.Dense {
	var res mat.Dense
	res.Apply(func(i, j int, v float64) float64 { return math.Exp(v) }, m)
	res.Scale(1.0/mat.Sum(&res), &res)
	return &res
}

func RandomMatrix(R, C int, factor float64) *mat.Dense {
	m := mat.NewDense(R, C, nil)
	for r := 0; r < R; r++ {
		for c := 0; c < C; c++ {
			m.Set(r, c, rand.NormFloat64()*factor)
		}
	}
	return m
}

func RandomIndices(N int) []int {
	arr := make([]int, N)
	for i := 0; i < N; i++ {
		arr[i] = i
	}
	rand.Shuffle(len(arr), func(i, j int) { arr[i], arr[j] = arr[j], arr[i] })
	return arr
}

func ReadDataset(filename string) *mat.Dense {
	var d, N int

	f, err := os.Open(filename)
	if err != nil {
		panic("")
	}
	defer f.Close()
	reader := csv.NewReader(f)

	data := make([]float64, 0)
	for {
		row, err := reader.Read()
		if err == io.EOF {
			N = len(data) / d
			break
		}
		floatrow := make([]float64, len(row))
		for i, s := range row {
			f, _ := strconv.ParseFloat(s, 64)
			floatrow[i] = f
		}
		d = len(floatrow)
		data = append(data, floatrow...)
	}

	m := mat.NewDense(N, d, data)
	return m
}

func Argmax(v mat.Vector) int {
	idx := 0
	mx := v.AtVec(idx)
	for i := 1; i < v.Len(); i++ {
		if v.AtVec(i) > mx {
			idx = i
			mx = v.AtVec(i)
		}
	}
	return idx
}
