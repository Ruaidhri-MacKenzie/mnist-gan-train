import { imageWidth, imageHeight, imageChannels } from "./config.js";
import { displayEpochReport, displayTrainingInfo } from "./ui.js";
import { selectRealSamples, generateFakeSamples } from "./data.js";

export const createDiscriminator = () => {
	const model = tf.sequential();
	model.add(tf.layers.conv2d({ filters: 64, kernelSize: [3, 3], strides: [2, 2], padding: "same", inputShape: [28, 28, 1] }));
	model.add(tf.layers.leakyReLU(0.2));
	model.add(tf.layers.dropout(0.4));
	model.add(tf.layers.conv2d({ filters: 64, kernelSize: [3, 3], strides: [2, 2], padding: "same" }));
	model.add(tf.layers.leakyReLU(0.2));
	model.add(tf.layers.dropout(0.4));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

	// Discriminator model is compiled separately as it is only trained independently of the combined model
	const learningRate = 0.0002;
	const beta1 = 0.5;
	const optimizer = tf.train.adam(learningRate, beta1);
	model.compile({ loss: "binaryCrossentropy", optimizer, metrics: ["accuracy"] });
	return model;
};

export const createGenerator = () => {
	const model = tf.sequential();

	// Encoder
	model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', inputShape: [28, 28, 1] }));
	model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }));
	model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }));
	model.add(tf.layers.flatten());
	
	// Latent vector
	model.add(tf.layers.dense({ units: 7 * 7 * 128, activation: 'relu' }));
	model.add(tf.layers.reshape({ targetShape: [7, 7, 128] }));
	
	// Output layer
	model.add(tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }));
	model.add(tf.layers.conv2dTranspose({ filters: 1, kernelSize: 3, strides: 2, padding: 'same', activation: 'tanh' }));
	// model.summary();

	// Generator model is not compiled as it is only trained as part of the combined model
	return model;
};

export const createGan = (generator, discriminator) => {
	// Discriminator weights are frozen while the combined model trains
	// This allows the generator to apply the gradient of the discriminator
	discriminator.trainable = false;

	const model = tf.sequential();
	model.add(generator);
	model.add(discriminator);

	// The GAN is compiled with a loss function and optimizer
	const learningRate = 0.0002;
	const beta1 = 0.5;
	const optimizer = tf.train.adam(learningRate, beta1);
	model.compile({ loss: "binaryCrossentropy", optimizer });

	return model;
};

export const downloadModel = (model, name) => {
	model.save(`downloads://${name}`);
};

export const trainModel = async (generator, discriminator, gan, dataset, epochs = 10, batchSize = 64) => {
	const batchesPerEpoch = Math.floor(dataset.shape[0] / batchSize);
	const halfBatch = Math.floor(batchSize / 2);

	for (let epoch = 0; epoch < epochs; epoch++) {
		for (let batch = 0; batch < batchesPerEpoch; batch++) {
			// Select half batch of real samples
			const [realSamples, realLabels] = selectRealSamples(dataset, halfBatch);

			// Generate half batch of fake samples
			const [fakeSamples, fakeLabels] = await generateFakeSamples(generator, halfBatch);

			// Merge real and fake samples to create training set for the discriminator
			const samples = tf.concat([realSamples, fakeSamples], 0);
			const labels = tf.concat([realLabels, fakeLabels], 0);

			// Train the discriminator, then freeze weights while generator trains
			discriminator.trainable = true;
			const [discLoss, discAcc] = await discriminator.trainOnBatch(samples, labels);
			discriminator.trainable = false;

			// Generate normally distributed noise as input for the generator
			const noise = tf.randomNormal([batchSize, imageHeight, imageWidth, imageChannels]);
			
			// Create real class labels for the fake samples
			const ganLabels = tf.ones([batchSize, 1]);
			
			// Train the generator using the discriminator's gradients
			const genLoss = await gan.trainOnBatch(noise, ganLabels);

			// Display training info for this batch
			displayTrainingInfo(epoch, batch, batchesPerEpoch, discLoss, genLoss);
			console.log(`Epoch: ${epoch + 1}, Batch: ${batch + 1}/${batchesPerEpoch}, Discriminator Loss: ${discLoss.toFixed(3)}, Generator Loss: ${genLoss.toFixed(3)}`);
		}
		// Display discriminator accuracy
		if ((epoch + 1) % 1 == 0) {
			displayEpochReport(epoch, generator, discriminator, dataset);
		}
	}
};

export const evaluateDiscriminator = (discriminator, X, y) => {
	const [loss, accuracy] = discriminator.evaluate(X, y, { verbose: 0 });
	return [loss, accuracy];
};
