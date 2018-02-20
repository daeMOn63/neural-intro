package main

import (
	"fmt"
	"math"
)

type Neuron struct {
	Weight     []float64
	Activation func(float64) float64
}

func (p Neuron) Predict(I1, I2, B float64) float64 {

	var output float64

	for i, input := range []float64{I1, I2, B} {
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

	var a, b float64

	a = 0.0
	b = 0.0
	fmt.Printf("(%f,%f) = %f\n", a, b, p.Predict(a, b, 1))

	a = 1.0
	b = 0.0
	fmt.Printf("(%f,%f) = %f\n", a, b, p.Predict(a, b, 1))

	a = 0.0
	b = 1.0
	fmt.Printf("(%f,%f) = %f\n", a, b, p.Predict(a, b, 1))

	a = 1.0
	b = 1.0
	fmt.Printf("(%f,%f) = %f\n", a, b, p.Predict(a, b, 1))

}
