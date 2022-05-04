import { Matrix, Dense } from './matrix_speedup.js';

export class Perceptron {
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    static sigmoidDifferential(y) {
        return y * (1 - y);
    }

    static ReLU(x) {
        return Math.max(0, x);
    }

    static ReLUDifferential(y) {
        return y < 0 ? 0 : 1;
    }

    /**
     * Creates Linear Neuronet
     * @param {Array<Number>} layerNumbers describes number of layers and number of neurons on each layer
     * @param {Function} neuronFunction non linear function wich process sum in each neurons
     */
    constructor(layerNumbers, startLearningRate, neuronFunction = { func: Perceptron.ReLU, differential: Perceptron.ReLUDifferential }) {
        this.neuronFunction = neuronFunction;

        this.learningRate = startLearningRate;

        this.layers = Array(layerNumbers.length);
        this.connections = Array(this.layers.length - 1);

        let connectionsTotal = 0, i;
        for (i = 0; i < layerNumbers.length - 1; i++) {
            this.layers[i] = Array(layerNumbers[i]);

            let connections = new Matrix(layerNumbers[i], layerNumbers[i + 1]);
            this.connections[i] = connections;

            connectionsTotal += layerNumbers[i] * layerNumbers[i + 1];
        }
        this.layers[i] = Array(layerNumbers[i]);

        this.output = Array(layerNumbers[i]);

        console.log('connections number:', connectionsTotal);
    }

    randomConnections() {
        let conns = this.connections,
            connsLen = conns.length;
        for (let l = 0; l < connsLen; l++)
            Matrix.fill(conns[l], () => Math.random() / 10);
    }

    fillConnections(connectionsWeights) {
        let conns = this.connections,
            connsLen = conns.length;
        for (let l = 0; l < connsLen; l++)
            Matrix.fill(conns[l], (i, j) => connectionsWeights[l][i][j]);
    }

    input(dataSet) {
        if (dataSet.length != this.layers[0].length) throw new Error('Input data has another size');
        this.layers[0] = dataSet;

        let neuronFunction = this.neuronFunction.func,
            layers = this.layers, layersLen = layers.length,
            conns = this.connections;

        for (let i = 0; i < layersLen - 1; i++) {
            let nextLayerResult = Dense.mul(layers[i], conns[i]),
                nextLayerResultLen = nextLayerResult.length,
                targetLayer = layers[i + 1];
            for (let j = 0; j < nextLayerResultLen; j++)
                targetLayer[j] = neuronFunction(nextLayerResult[j]);
        }

        // Softmax for output layer
        let lastLayer = this.layers[this.layers.length - 1],
            lastLayerLen = lastLayer.length,
            lastLayerSum = 0;

        for (let res of lastLayer) lastLayerSum += res;

        if (lastLayerSum != 0)
            for (let i = 0; i < lastLayerLen; i++) {
                this.output[i] = lastLayer[i] / lastLayerSum;
            }
        //else
        //    process.stdout.write('! 0 SUM; ');

        return this.output;
    }

    killNaN(value, name) {
        if (this.checkNaN(value, name)) {
            process.exit();
        }
    }

    checkNaN(value, name) {
        if (Number.isNaN(value)) {
            console.log('is NaN:', name);
            return true;
        }
        return false;
    }

    train(input, expected) {
        let output = this.input(input), outputLength = output.length,
            output_E = Array(output.length); //output layer errors

        let error = 0;
        for (let i = 0; i < outputLength; i++) error += (expected[i] - output[i]) ** 2;
        error /= 2;

        // output layer errors
        for (let i = 0; i < outputLength; i++)
            output_E[i] = -(expected[i] - output[i]);

        let layers = this.layers, layersLength = layers.length,
            allConns = this.connections,
            dy = this.neuronFunction.differential, // differential function
            E = output_E, // errors on j layer
            gradient = 0,
            lr = this.learningRate;

        // all weights delta
        // i - curr layer, j - next layer : used to calc gradients on hidden and input layers
        for (let l = layersLength - 2; l > -1; --l) {
            let y = layers[l], // i layer
                Y = layers[l + 1], // j layer
                iN = y.length,
                jN = Y.length,
                w = allConns[l], // w - weights between i and j layers
                e = Array(iN); // errors on i layer
            for (let i = 0; i < iN; i++) { // each neuron/connection number in i layer
                let w_i = w[i]; e[i] = 0;
                for (let j = 0; j < jN; j++) { // each neuron/connection number in j layer
                    gradient = dy(Y[j]) * E[j];
                    e[i] += w_i[j] * gradient;
                    w_i[j] -= lr * y[i] * gradient;
                }
            }
            E = e;
        }
        if (Number.isNaN(E[0])) {
            console.log('NaN!!!');
            process.exit();
        }

        return error;
    }
};