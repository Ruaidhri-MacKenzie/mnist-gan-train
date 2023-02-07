import { selectRealSamples, generateFakeSamples, selectSample, sampleToImageData } from "./data.js";
import { evaluateDiscriminator } from "./model.js";

export const DOM = {
	startTraining: document.getElementById("start-training"),
	saveGenerator: document.getElementById("save-generator"),
	saveDiscriminator: document.getElementById("save-discriminator"),
	saveGan: document.getElementById("save-gan"),
	generateImage: document.getElementById("generate-image"),
	generateCanvas: document.getElementById("generate-canvas"),
	generatedImages: document.getElementById("generated-images"),
	discriminatorLoss: document.getElementById("discriminator-loss"),
	generatorLoss: document.getElementById("generator-loss"),
	currentEpoch: document.getElementById("current-epoch"),
	currentBatch: document.getElementById("current-batch"),
};

export const createImage = async (sample, scale = 1) => {
	const imageData = sampleToImageData(sample);
	const canvas = document.createElement("canvas");
	await tf.browser.toPixels(imageData, canvas);
	tf.dispose(sample);
	tf.dispose(imageData);

	const image = new Image();
	image.src = canvas.toDataURL();
	image.width = canvas.width * scale;
	image.height = canvas.height * scale;
	return image;
};

export const displayImage = async (sample, container, scale = 1) => {
	const image = await createImage(sample, scale);
	container.appendChild(image);
};

export const displayBatch = async (batch, container, scale = 1) => {
	const batchContainer = document.createElement("div");
	batchContainer.classList.add("batch-container");
	for (let i = 0; i < batch.shape[0]; i++) {
		const sample = selectSample(batch, i);
		const image = await createImage(sample, scale);
		batchContainer.appendChild(image);
	}
	container.appendChild(batchContainer);
	tf.dispose(batch);
};

export const displayTrainingInfo = (epoch, batch, batchesPerEpoch, discLoss, genLoss) => {
	DOM.currentEpoch.innerText = `Epoch: ${epoch + 1}`;
	DOM.currentBatch.innerText = `Batch: ${batch + 1}/${batchesPerEpoch}`;
	DOM.discriminatorLoss.innerText = `Discriminator Loss: ${discLoss.toFixed(3)}`;
	DOM.generatorLoss.innerText = `Generator Loss: ${genLoss.toFixed(3)}`;
};

export const displayAccuracy = (realAccuracy, fakeAccuracy) => {
	console.log(`Accuracy real: ${(realAccuracy * 100).toFixed(2)}%, fake: ${(fakeAccuracy * 100).toFixed(2)}%`);
};

export const displayGeneratedImages = async (batch, container, title) => {
	// Create batch container
	const batchContainer = document.createElement("div");
	batchContainer.classList.add("result");

	// Create batch heading
	const heading = document.createElement("h2");
	heading.innerText = title;
	batchContainer.appendChild(heading);

	// Display generated samples as images
	for (let i = 0; i < 5; i++) {
		const sample = selectSample(batch, i);
		const imageData = sampleToImageData(sample);
		const image = await createImage(imageData);
		batchContainer.appendChild(image);
	}

	container.appendChild(batchContainer);
};

export const displayEpochReport = async (epoch, generator, discriminator, dataset, batchSize = 100) => {
	// Prepare real and fake samples
	const [realSamples, realLabels] = selectRealSamples(dataset, batchSize);
	const [fakeSamples, fakeLabels] = await generateFakeSamples(generator, batchSize);

	// Evaluate discriminator accuracy on real and fake samples
	const [realLoss, realAccuracy] = evaluateDiscriminator(discriminator, realSamples, realLabels);
	const [fakeLoss, fakeAccuracy] = evaluateDiscriminator(discriminator, fakeSamples, fakeLabels);

	// Display the discriminator accuracy
	displayAccuracy(realAccuracy.dataSync(), fakeAccuracy.dataSync());

	// Display the generated images
	displayGeneratedImages(fakeSamples, DOM.generatedImages, `Epoch ${epoch + 1}:`);
};

export const generateToCanvas = async (generator, canvas) => {
	// Generate a fake sample
	const [samples, labels] = await generateFakeSamples(generator, 1);
	
	// Convert to image data
	const imageData = sampleToImageData(samples);
	
	// Render image data to canvas
	await tf.browser.toPixels(imageData, canvas);
};
