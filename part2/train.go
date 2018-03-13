package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Neuron struct {
	Weight               []float64
	Activation           func(float64) float64
	ActivationDerivative func(float64) float64
}

func (p Neuron) Predict(x1, x2 float64) float64 {

	var output float64

	for i, input := range []float64{x1, x2, 1} {
		output += p.Weight[i] * input
	}

	return p.Activation(output)
}

func (p Neuron) Train(inputs [][]float64, expectedOutputs []float64) {

	epochs := 1000      // Number of times we'll process all given inputs
	learningRate := 0.1 // The "speed" of the network to learn. Also seen as the length of the "jumps" the line will make between each epochs.

	for i := 0; i < epochs; i++ {

		for j, input := range inputs { // Loop on all inputs

			x1 := input[0]
			x2 := input[1]

			output := p.Predict(x1, x2)

			currentError := -2 * (expectedOutputs[j] - output) // E'(y) = -2 * (t - y)

			slope := p.ActivationDerivative(output) // A'(S) = y * (1 - y)

			x1Gradient := slope * currentError * x1 // dE/dw1 = -2 * (t - y) * (y * (1 - y)) * x1
			x2Gradient := slope * currentError * x2 // dE/dw2 = -2 * (t - y) * (y * (1 - y)) * x2
			bGradient := slope * currentError       // dE/dw2 = -2 * (t - y) * (y * (1 - y))

			p.Weight[0] = p.Weight[0] - x1Gradient*learningRate
			p.Weight[1] = p.Weight[1] - x2Gradient*learningRate
			p.Weight[2] = p.Weight[2] - bGradient*learningRate
		}

	}
}

func main() {

	randSource := rand.NewSource(time.Now().UnixNano())
	rng := rand.New(randSource)

	p := &Neuron{
		Weight: []float64{rng.Float64(), rng.Float64(), rng.Float64()},
		Activation: func(x float64) float64 {
			return 1.0 / (1.0 + math.Exp(-x))
		},
		ActivationDerivative: func(y float64) float64 {
			return y * (1.0 - y)
		},
	}

	p.Train(
		[][]float64{ // List of Inputs
			[]float64{0, 0},
			[]float64{0, 1},
			[]float64{1, 0},
			[]float64{1, 1},
		},
		[]float64{ // Expected outputs
			0,
			1,
			1,
			1,
		},
	)

	// Test our data and print the network result
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
