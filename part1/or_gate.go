package main

import (
	"fmt"
	"math"
)

type Neuron struct {
	Weight     []float64
	Activation func(float64) float64
}

func (p Neuron) Predict(x1, x2, b float64) float64 {

	var output float64

	for i, input := range []float64{x1, x2, b} {
		output += p.Weight[i] * input
	}
	return p.Activation(output)
}

func main() {

	p := &Neuron{
		Weight: []float64{400.0, 600.0, -100.0},
		Activation: func(x float64) float64 {
			return 1.0 / (1.0 + math.Exp(-x))
		},
	}

	var x1, x2 float64

	x1 = 0.0
	x2 = 0.0
	fmt.Printf("(%f,%f) = %f\n", x1, x2, p.Predict(x1, x2, 1))

	x1 = 1.0
	x2 = 0.0
	fmt.Printf("(%f,%f) = %f\n", x1, x2, p.Predict(x1, x2, 1))

	x1 = 0.0
	x2 = 1.0
	fmt.Printf("(%f,%f) = %f\n", x1, x2, p.Predict(x1, x2, 1))

	x1 = 1.0
	x2 = 1.0
	fmt.Printf("(%f,%f) = %f\n", x1, x2, p.Predict(x1, x2, 1))
}
