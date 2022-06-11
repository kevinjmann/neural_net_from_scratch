const canvas = document.getElementById('drawing-board');
const clearBtn = document.getElementById('btn-clear');
const guessBtn = document.getElementById('btn-guess');
const guessTxt = document.getElementById('txt-guess');
const ctx = canvas.getContext('2d');

const title = document.getElementById("title");
const inputSideLength = 28;
canvas.width = inputSideLength;
canvas.height = inputSideLength;
ctx.strokeStyle = 'black'
let isPainting = false;
let lineWidth = 2;

let startX;
let startY;

clearBtn.onclick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
}


canvas.addEventListener('mousedown', (e) => {
    isPainting = true;
    startX = e.clientX;
    startY = e.clientY;
    ctx.stroke();
})

canvas.addEventListener('mouseup', (e) => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();
});
const draw = (e) => {
    if(!isPainting) {
        return;
    }
    ctx.lineWidth = lineWidth;
    ctx.lineCape = 'round';
    ctx.lineTo(e.clientX - 8, e.clientY - 8);
    ctx.stroke();
}
canvas.addEventListener('mousemove', draw)
guessBtn.onclick = () => {
    let inputData = new Module.VectorFloat();
    let imData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for(let i = 3; i < imData.length; i+=4) {
        inputData.push_back(imData[i]/255.0);
    }
    guessTxt.innerText = Module.doWasmInference(inputData);
}
