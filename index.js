import { Perceptron } from "./logic/perceptron.js";

async function main() {
    let buttons = {
        clear: document.getElementById('clear-button'),
        recognise: document.getElementById('recognise-button'),
    };

    buttons.clear.addEventListener('click', e => {
        paper.clearRect(0, 0, paperElement.width, paperElement.height);
    });
    buttons.recognise.addEventListener('click', e => {
        recognise();
    });

    /** @type {HTMLCanvasElement} */
    let paperElement = document.getElementById('paper');
    let paper = paperElement.getContext('2d');
    paper.lineWidth = 25;
    paper.lineCap = 'round';
    paper.lineJoin = 'round';
    paper.ondown = e => {
        paper.prevX = paper.prevY = 0;
        paper.drawE(e);
    };
    paper.drawE = e => {
        let rect = paperElement.getBoundingClientRect();
        let x = e.clientX, y = e.clientY;
        if (!(x <= rect.right && x >= rect.left && y >= rect.top && y <= rect.bottom)) return;
        else x -= rect.left; y -= rect.top;
        if (screen.width < 500) {
            x *= 280 / 200;
            y *= 280 / 200;
        }
        paper.drawLineTo(x, y);
    };

    paperElement.addEventListener('mousedown', e => {
        paper.ondown(e);
        paper.touched = true;
    });
    paperElement.addEventListener('mousemove', e => {
        if (paper.touched)
            paper.drawE(e);
    });
    paperElement.addEventListener('mouseup', e => {
        paper.touched = false;
    });

    paperElement.addEventListener('touchstart', e => paper.ondown(e.touches[0]));
    paperElement.addEventListener('touchmove', e => {
        e.preventDefault();
        paper.drawE(e.touches[0]);
    });

    paper.drawLineTo = function (x, y) {
        paper.prevX;
        paper.beginPath();
        paper.moveTo(paper.prevX || x, paper.prevY || y);
        paper.lineTo(x, y);
        paper.closePath();
        paper.stroke();
        paper.prevX = x; paper.prevY = y;
    };


    let connections = await fetch('./connections.json').then(res => res.json());
    let perceptron = new Perceptron([784, 200, 10], 0.01);
    perceptron.fillConnections(connections);


    function recognise() {
        let imgData = paper.getImageData(0, 0, 280, 280);
        imgUtil = window.imgUtil;

        let grayscaleImg = imgUtil.imageDataToGrayscale(imgData);
        let boundingRectangle = imgUtil.getBoundingRectangle(grayscaleImg, 0.01);
        let trans = imgUtil.centerImage(grayscaleImg); // [dX, dY] to center of mass

        let canvasCopy = document.createElement("canvas");
        canvasCopy.width = imgData.width;
        canvasCopy.height = imgData.height;
        let copyCtx = canvasCopy.getContext("2d");
        let brW = boundingRectangle.maxX + 1 - boundingRectangle.minX;
        let brH = boundingRectangle.maxY + 1 - boundingRectangle.minY;
        let scaling = 190 / (brW > brH ? brW : brH);
        // scale
        copyCtx.translate(paperElement.width / 2, paperElement.height / 2);
        copyCtx.scale(scaling, scaling);
        copyCtx.translate(-paperElement.width / 2, -paperElement.height / 2);
        // translate to center of mass
        copyCtx.translate(trans.transX, trans.transY);

        copyCtx.drawImage(paper.canvas, 0, 0);

        // now bin image into 10x10 blocks (giving a 28x28 image)
        imgData = copyCtx.getImageData(0, 0, 280, 280);
        grayscaleImg = imgUtil.imageDataToGrayscale(imgData);

        let inputData = new Array(784);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                let mean = 0;

                for (let v = 0; v < 10; v++)
                    for (let h = 0; h < 10; h++)
                        mean += grayscaleImg[y * 10 + v][x * 10 + h];

                inputData[x * 28 + y] = (mean / 100); // average and invert
            }
        }

        // painting input to the paper.
        let input = [];
        paper.clearRect(0, 0, paperElement.width, paperElement.height);
        paper.drawImage(copyCtx.canvas, 0, 0);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                let block = paper.getImageData(x * 10, y * 10, 10, 10);
                let newVal = 255 * (inputData[x * 28 + y]);
                input.push(Math.round((1 - inputData[x * 28 + y]) * 100) / 100);
                for (let i = 0; i < 4 * 10 * 10; i += 4) {
                    block.data[i] = newVal;
                    block.data[i + 1] = newVal;
                    block.data[i + 2] = newVal;
                    block.data[i + 3] = 255;
                }
                paper.putImageData(block, x * 10, y * 10);
            }
        }

        let output = perceptron.input(input);
        let maxi = 0, max = output[0];
        for (let i = 0; i < output.length; i++) {
            if (output[i] > max) {
                max = output[i]; maxi = i;
            }
        }
        document.getElementById('result').innerText = maxi.toString();
    }
}

main();