package main

import (
	"log"
	"math/rand"
	"neural"
	"neural/utils"
	"runtime"
	"time"
)

const (
	HIDDEN = 100
	OUT    = 10
	LR     = 0.01
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	xtrain := utils.ReadDataset("/home/julius/code/golang/neural/data/xtrain.csv")
	ytrain := utils.ReadDataset("/home/julius/code/golang/neural/data/ytrain.csv")
	xtest := utils.ReadDataset("/home/julius/code/golang/neural/data/xtest.csv")
	ytest := utils.ReadDataset("/home/julius/code/golang/neural/data/ytest.csv")

	log.Println("Datasets read")
	_, d := xtrain.Dims()
	nn := neural.NewNN(LR, OUT, d, HIDDEN)

	nn.Train(xtrain, ytrain, 3)

	nn.Test(xtest, ytest)

}
