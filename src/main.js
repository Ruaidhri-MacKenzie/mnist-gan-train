import { epochs, batchSize } from "./config.js";
import { loadDataset } from "./data.js";
import { createGenerator, createDiscriminator, createGan, downloadModel, trainModel } from "./model.js";
import { DOM, generateToCanvas } from "./ui.js";

const dataset = await loadDataset();

const discriminator = createDiscriminator();
const generator = createGenerator();
const gan = createGan(generator, discriminator);

DOM.startTraining.addEventListener("click", (event) => {
	trainModel(generator, discriminator, gan, dataset, epochs, batchSize);
});

DOM.saveGenerator.addEventListener("click", (event) => {
	downloadModel(generator, "mnist-generator");
});

DOM.saveDiscriminator.addEventListener("click", (event) => {
	downloadModel(discriminator, "mnist-discriminator");
});

DOM.saveGan.addEventListener("click", (event) => {
	downloadModel(gan, "mnist-gan");
});

DOM.generateImage.addEventListener("click", (event) => {
	generateToCanvas(generator, DOM.generateCanvas);
});
