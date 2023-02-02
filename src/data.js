import { MnistData } from "./mnist.js";

let batch = 0;
let index = 0;

export const selectRealSamples = (dataset, batchSize) => {
	// Select random batch of contiguous instances
	// const index = Math.floor(Math.random() * (dataset.shape[0] - batchSize));
	// const X = dataset.slice([index, 0, 0, 0], [batchSize, 28, 28, 1]);

	// Select first batch of instances
	// const X = dataset.slice([0, 0, 0, 0], [batchSize, 28, 28, 1]);

	const batchesPerEpoch = Math.floor(dataset.shape[0] / batchSize);
	const X = dataset.slice([index, 0, 0, 0], [batchSize, 28, 28, 1]);
	batch = (batch + 1) % batchesPerEpoch;
	index = batch * batchSize;

	// generate "real" class labels (1)
	const y = tf.ones([batchSize, 1]);

	return [X, y];
};

export const generateFakeSamples = async (generator, latentDim, batchSize) => {
	// generate points in latent space
	const xs = tf.randomNormal([batchSize, latentDim]);

	// predict outputs
	const X = await generator.predict(xs);
	
	// create 'fake' class labels (0)
	const y = tf.zeros([batchSize, 1]);
	
	return [X, y];
};

export const loadRealSamples = async () => {
	// load mnist dataset
	const mnist = new MnistData();
	await mnist.load();
	const trainData = mnist.getTrainData();
  // const testData = mnist.getTestData();
	return trainData.xs;
};
