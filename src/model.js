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

export const createGenerator = (latentDim) => {
	const model = tf.sequential();

	// 7x7 image
	model.add(tf.layers.dense({ units: 128 * 7 * 7, inputDim: latentDim }));
	model.add(tf.layers.leakyReLU(0.2));
	model.add(tf.layers.reshape({ targetShape: [7, 7, 128] }));
	
	// upsample to 14x14
	model.add(tf.layers.conv2dTranspose({ filters: 128, kernelSize: [4, 4], strides: [2, 2], padding: "same" }));
	model.add(tf.layers.leakyReLU(0.2));
	
	// upsample to 28x28
	model.add(tf.layers.conv2dTranspose({ filters: 128, kernelSize: [4, 4], strides: [2, 2], padding: "same" }));
	model.add(tf.layers.leakyReLU(0.2));

	model.add(tf.layers.conv2d({ filters: 1, kernelSize: [7, 7], activation: "sigmoid", padding: "same" }));

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

export const trainModel = async (generator, discriminator, gan, dataset, latentDim, epochs = 10, batchSize = 64) => {
	const batchesPerEpoch = Math.floor(dataset.shape[0] / batchSize);
	const halfBatch = Math.floor(batchSize / 2);

	for (let epoch = 0; epoch < epochs; epoch++) {
		for (let batch = 0; batch < batchesPerEpoch; batch++) {
			// get randomly selected 'real' samples
			const [xReal, yReal] = selectRealSamples(dataset, halfBatch);

			// generate 'fake' examples
			const [xFake, yFake] = await generateFakeSamples(generator, latentDim, halfBatch);

			// create training set for the discriminator
			const X = tf.concat([xReal, xFake], 0);
			const y = tf.concat([yReal, yFake], 0);

			// update discriminator model weights
			discriminator.trainable = true;
			const [discLoss, discAcc] = await discriminator.trainOnBatch(X, y);
			discriminator.trainable = false;

			let genLoss;
			for (let i = 0; i < 5; i++) {
				// prepare points in latent space as input for the generator
				const xGan = tf.randomNormal([batchSize, latentDim]);
				
				// create inverted labels for the fake samples
				const yGan = tf.ones([batchSize, 1]);
				
				// update the generator via the discriminator's error
				genLoss = await gan.trainOnBatch(xGan, yGan);
			}

			// summarize loss on this batch
			displayTrainingInfo(epoch, batch, batchesPerEpoch, discLoss, genLoss);
			console.log(`Epoch: ${epoch + 1}, Batch: ${batch + 1}/${batchesPerEpoch}, Discriminator Loss: ${discLoss.toFixed(3)}, Generator Loss: ${genLoss.toFixed(3)}`);
		}
		// evaluate the model performance, sometimes
		if ((epoch + 1) % 1 == 0) {
			displayEpochReport(epoch, generator, discriminator, dataset, latentDim);
		}
	}
};

export const evaluateDiscriminator = (discriminator, X, y) => {
	const [loss, accuracy] = discriminator.evaluate(X, y, { verbose: 0 });
	return [loss, accuracy];
};
