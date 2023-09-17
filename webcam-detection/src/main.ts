import * as tf from '@tensorflow/tfjs';
import { CLASSES } from './classes'
import "core-js/stable";
import "regenerator-runtime/runtime";

async function detect(model: tf.GraphModel, video: HTMLVideoElement, webcam: [CanvasRenderingContext2D, number, number]) {
    const [ctx, imgHeight, imgWidth] = webcam;
    const myTensor = tf.browser.fromPixels(video);

    const readyfied = tf.expandDims(myTensor, 0);
    const results = await model.executeAsync(readyfied);

    const detectionThreshold = 0.4;
    const iouThreshold = 0.5;
    const maxBoxes = 20;
    const prominentDetection = tf.topk(results[0]);
    const justBoxes = results[1].squeeze();
    const justValues = prominentDetection.values.squeeze();

    const [maxIndices, scores, boxes] = await Promise.all([
        prominentDetection.indices.data(),
        justValues.array(),
        justBoxes.array(),
    ]);

    const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(
        justBoxes,
        justValues,
        maxBoxes,
        iouThreshold,
        detectionThreshold,
        1,
    );

    const chosen = await nmsDetections.selectedIndices.data();

    tf.dispose([
        results[0],
        results[1],
        nmsDetections.selectedIndices,
        nmsDetections.selectedScores,
        prominentDetection.indices,
        prominentDetection.values,
        myTensor,
        readyfied,
        justBoxes,
        justValues,
    ]);

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    chosen.forEach((detection) => {
        ctx.strokeStyle = '#0F0';
        ctx.lineWidth = 4;
        ctx.globalCompositeOperation = 'destination-over';
        const detectedIndex = maxIndices[detection];
        const detectedClass = CLASSES[detectedIndex];
        const detectedScore = scores[detection];
        const dBox = boxes[detection];

        const startY = dBox[0] > 0 ? dBox[0] * imgHeight : 0;
        const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0;
        const height = (dBox[2] - dBox[0]) * imgHeight;
        const width = (dBox[3] - dBox[1]) * imgWidth;
        ctx.strokeRect(startX, startY, width, height);
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = '#0B0';
        const textHeight = 16;
        const textPad = 4;
        const label = `Recognized: ${detectedClass} ${Math.round(detectedScore * 100)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(startX, startY, textWidth + textPad, textHeight + textPad);
        ctx.fillStyle = '#000000';
        ctx.fillText(label, startX, startY);
    });

    requestAnimationFrame(() => {
        detect(model, video, webcam);
    });
}

async function setupWebcam(video: HTMLVideoElement): Promise<[CanvasRenderingContext2D, number, number]> {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw Error('Missing webcam!');
    }

    video.srcObject = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user' },
    });

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            const cvs = document.querySelector<HTMLCanvasElement>('#detection');
            const ctx = cvs.getContext('2d');
            const imgWidth = video.clientWidth;
            const imgHeight = video.clientHeight;
            cvs.width = imgWidth;
            cvs.height = imgHeight;
            ctx.font = '16px sans-serif';
            ctx.textBaseline = 'top';
            resolve([ctx, imgHeight, imgWidth]);
        };
    });
}

async function loadModel() {
    await tf.ready();
    const modelPath = 'https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1';
    return await tf.loadGraphModel(modelPath, { fromTFHub: true });
}

async function bootstrap() {
    const model = await loadModel();
    const video = document.querySelector<HTMLVideoElement>('#video')!;
    const webcam = await setupWebcam(video);
    detect(model, video, webcam);
}

bootstrap();
