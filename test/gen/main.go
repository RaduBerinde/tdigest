package main

import (
	"math/rand/v2"
	"os"
	"strconv"
)

const (
	N     = 1e6
	Mu    = 10
	Sigma = 3

	seed = 42
)

func main() {
	// Generate uniform and normal data
	rng := rand.New(rand.NewPCG(seed, seed))
	uniformData := make([]float64, N)
	for i := range uniformData {
		uniformData[i] = rng.Float64()
	}

	rng = rand.New(rand.NewPCG(seed, seed))
	normalData := make([]float64, N)
	for i := range normalData {
		normalData[i] = rng.NormFloat64()*Sigma + Mu
	}

	smallData := []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}

	writeData("uniform.dat", uniformData)
	writeData("normal.dat", normalData)
	writeData("small.dat", smallData)
}

func writeData(name string, data []float64) {
	f, err := os.Create(name)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	buf := make([]byte, 0, 64)
	for _, x := range data {
		buf = strconv.AppendFloat(buf, x, 'f', -1, 64)
		_, err := f.Write(buf)
		if err != nil {
			panic(err)
		}
		_, err = f.Write([]byte{'\n'})
		if err != nil {
			panic(err)
		}
		buf = buf[0:0]
	}
}
