import { imageWidth, imageHeight, imageChannels } from "./config.js";
import { MnistData } from "./mnist.js";

let batchNum = 0;
let index = 0;

export const loadDataset = async () => {
	const mnist = new MnistData();
	await mnist.load();
	return tf.tidy(() => {
		const trainData = mnist.getTrainData();
		return trainData.xs;
	});
};

export const selectSample = (dataset, index = 0) => {
	return tf.tidy(() => {
		const size = dataset.shape[0];
		if (index >= size) index = size - 1;
		const batch = dataset.slice([index, 0, 0, 0], [1, imageHeight, imageWidth, imageChannels]);
		const sample = batch.reshape([imageHeight, imageWidth, imageChannels]);
		return sample;
	});
};

export const selectBatch = (dataset, batchIndex, batchSize) => {
	return tf.tidy(() => {
		const size = dataset.shape[0];
		let index = batchIndex * batchSize;
		if (index >= size - batchSize) index = size - batchSize - 1;
		const batch = dataset.slice([index, 0, 0, 0], [batchSize, imageHeight, imageWidth, imageChannels]);
		return batch;
	});
};

export const generateNoise = (inputShape) => {
	return tf.tidy(() => {
		const noise = tf.randomNormal(inputShape);
		const shiftedNoise = tf.add(noise, 0.5);
		const clippedNoise = tf.clipByValue(shiftedNoise, 0, 1);
		return clippedNoise;
	});
};

export const selectRealSamples = (dataset, batchSize = 1) => {
	// Select batch of real samples
	const batch = selectBatch(dataset, index, batchSize);

	// Increase index so that next batch is selected
	const batchesPerEpoch = Math.floor(dataset.shape[0] / batchSize);
	batchNum = (batchNum + 1) % batchesPerEpoch;
	index = batchNum * batchSize;

	// Create real class labels
	const labels = tf.ones([batchSize, 1]);

	return [batch, labels];
};

export const generateFakeSamples = async (generator, batchSize = 1) => {
	// Generate normally distributed noise
	const noise = tf.randomNormal([batchSize, imageHeight, imageWidth, imageChannels]);

	// Generate fake sample from noise
	const batch = await generator.predict(noise);
	
	// Create fake class labels
	const labels = tf.zeros([batchSize, 1]);
	
	return [batch, labels];
};

export const sampleToImageData = (sample) => {
	return sample.reshape([imageHeight, imageWidth, imageChannels]).add(1).mul(0.5);
};
