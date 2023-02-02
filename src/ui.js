import { selectRealSamples, generateFakeSamples } from "./data.js";
import { evaluateDiscriminator } from "./model.js";

export const DOM = {
	startTraining: document.getElementById("start-training"),
	saveGenerator: document.getElementById("save-generator"),
	saveDiscriminator: document.getElementById("save-discriminator"),
	saveGan: document.getElementById("save-gan"),
	generateImage: document.getElementById("generate-image"),
	generateCanvas: document.getElementById("generate-canvas"),
	discriminatorLoss: document.getElementById("discriminator-loss"),
	generatorLoss: document.getElementById("generator-loss"),
	currentEpoch: document.getElementById("current-epoch"),
	currentBatch: document.getElementById("current-batch"),
};

export const displayTrainingInfo = (epoch, batch, batchesPerEpoch, discLoss, genLoss) => {
	DOM.currentEpoch.innerText = `Epoch: ${epoch + 1}`;
	DOM.currentBatch.innerText = `Batch: ${batch + 1}/${batchesPerEpoch}`;
	DOM.discriminatorLoss.innerText = `Discriminator Loss: ${discLoss.toFixed(3)}`;
	DOM.generatorLoss.innerText = `Generator Loss: ${genLoss.toFixed(3)}`;
};

export const displayAccuracy = (accReal, accFake) => {
	console.log(`Accuracy real: ${(accReal * 100).toFixed(2)}%, fake: ${(accFake * 100).toFixed(2)}%`);
};

export const displayGeneratedImage = async (xs, epoch) => {
	const container = document.createElement("div");
	container.classList.add("result");
	const title = document.createElement("h2");
	title.innerText = `Epoch ${epoch + 1}:`;

	xs = xs.slice([0, 0, 0, 0], [1, 28, 28, 1]).reshape([28, 28, 1]);
	const canvas = document.createElement("canvas");
	await tf.browser.toPixels(xs, canvas);
	
	container.appendChild(title);
	container.appendChild(canvas);
	const generatedImages = document.getElementById("generated-images");
	generatedImages.appendChild(container);
};

export const displayEpochReport = async (epoch, generator, discriminator, dataset, latentDim, batchSize = 100) => {
	// prepare real samples
	const [xReal, yReal] = selectRealSamples(dataset, batchSize);

	// evaluate discriminator on real examples
	const [lossReal, accReal] = evaluateDiscriminator(discriminator, xReal, yReal);

	// prepare fake examples
	const [xFake, yFake] = await generateFakeSamples(generator, latentDim, batchSize);

	// evaluate discriminator on fake examples
	const [lossFake, accFake] = evaluateDiscriminator(discriminator, xFake, yFake);

	// summarize discriminator performance
	displayAccuracy(accReal.dataSync(), accFake.dataSync());

	// save plot
	displayGeneratedImage(xFake, epoch);
};

export const generateToCanvas = async (generator, latentDim, canvas) => {
	const [X, y] = await generateFakeSamples(generator, latentDim, 1);
	await tf.browser.toPixels(X.reshape([28, 28, 1]), canvas);
};
