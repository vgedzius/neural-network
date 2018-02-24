const NeuralNetwork = require('./neural-network');

const LAYERS = [2, 4, 4, 1];
const POOLSIZE = 20;
const GENERATIONS = 5000;

let pool = [];
let trainingData = [
  {
    inputs: [0, 1],
    target: [1],
  },
  {
    inputs: [1, 0],
    target: [1],
  },
  {
    inputs: [0, 0],
    target: [0],
  },
  {
    inputs: [1, 1],
    target: [0],
  },
];

for (let i = 0; i < POOLSIZE; i++) {
  let n = new NeuralNetwork(LAYERS);

  n.calculateFitness(trainingData);
  
  pool.push(n);
}

for (let i = 0; i < GENERATIONS; i++) {
  pool = pool.sort((a, b) => a.fitness - b.fitness)
    .map((network, index) => {
      if (index >= POOLSIZE / 2) {
        let r = [];
        r.push(network.clone());
        r.push(network.mutate());
        return r;
      }
    }).filter((item) => item !== undefined)
    .reduce((a, b) => {
      return a.concat(b);
    }).map((n) => n.calculateFitness(trainingData));
}

let best = pool.pop();

trainingData.forEach((data) => {
  console.log(`Input: ${data.inputs}, Output: ${Math.round(Math.abs(best.feedForward(data.inputs)))}`);
});
