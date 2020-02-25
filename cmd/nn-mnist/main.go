package main

import (
	"fmt"
	"log"
	"math/rand"
	"neural"
	"neural/utils"
	"os"
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

	datadir := os.Getenv("DATADIR")
	if datadir == "" {
		fmt.Println("Set ENV variable DATADIR before running")
		os.Exit(0)
	}

	args := os.Args[1:]

	xtrain := utils.ReadDataset(datadir + "/" + "x" + args[0])
	ytrain := utils.ReadDataset(datadir + "/" + "y" + args[0])
	xtest := utils.ReadDataset(datadir + "/" + "x" + args[1])
	ytest := utils.ReadDataset(datadir + "/" + "y" + args[1])

	log.Println("Datasets read")
	_, d := xtrain.Dims()
	nn := neural.NewNN(LR, OUT, d, HIDDEN)

	nn.Train(xtrain, ytrain, 3)

	nn.Test(xtest, ytest)

}
