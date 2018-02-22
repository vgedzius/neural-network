class NeuralNetwork {
  constructor(layers) {
    this.layers = layers;
    this.neurons = [];
    this.weights = [];

    this.initNeurons();
    this.initWeights();
  }

  initNeurons() {
    this.layers.forEach((n) => {
      let layer = new Array(n).fill(0);
      this.neurons.push(layer);
    });
  }

  initWeights() {
    for (let i = 1; i < this.layers.length; i++) {
      let layersWeights = [];
      const previousLayer = this.layers[i - 1];

      for (let j = 0; j < this.neurons[i].length; j++) {
        let neuronWeights = [];

        for (let k = 0; k < previousLayer; k++) {
          neuronWeights[k] = Math.random() - 0.5;
        }
        layersWeights.push(neuronWeights);
      }
      this.weights.push(layersWeights);
    }
  }

  feedForward(inputs) {
    inputs.forEach((n, i) => {
      this.neurons[0][i] = n;
    });

    for (let i = 1; i < this.layers.length; i++) {
      for (let j = 0; j < this.neurons[i].length; j++) {
        let value = 0.25;
        for (let k = 0; k < this.neurons[i - 1].length; k++) {
          value += this.weights[i - 1][j][k] * this.neurons[i - 1][k];
        }
        this.neurons[i][j] = Math.tanh(value);
      }
    }

    return this.neurons[this.neurons.length - 1];
  }

  mutate() {
    this.weights = this.weights.map((layer) => {
      return layer.map((neurons) => {
        return neurons.map((weight) => {
          let random = Math.random() * 1000;

          if (random < 2) {
            return weight * -1;
          } else if (random < 4) {
            return Math.random() - 0.5;
          } else if (random < 6) {
            let factor = Math.random() + 1;
            return weight * factor;
          } else if (random < 8) {
            let factor = Math.random();
            return weight * factor;
          }
        });
      });
    });
  }
}