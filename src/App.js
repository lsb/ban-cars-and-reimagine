import React, { useRef, useEffect } from 'react';
import './App.css';
import Webcam from 'react-webcam';
import { AutoModel, pipeline, env, RawImage, CLIPTextModelWithProjection, AutoTokenizer, SegformerFeatureExtractor } from '@xenova/transformers';
import * as ort from 'onnxruntime-web';

env.backends.onnx.wasm.wasmPaths = window.location.href.replace(/[^/]+$/, "") + "static/js/";

const Canvas = props => {
  
  const { draw, ...rest } = props
  const canvasRef = useRef(null)
  
  useEffect(() => {
    
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    let animationFrameId
    
    const render = () => {
      draw(context)
      animationFrameId = window.requestAnimationFrame(render)
    }
    render()
    
    return () => {
      window.cancelAnimationFrame(animationFrameId)
    }
  }, [draw])
  
  return <canvas ref={canvasRef} {...rest}/>
} // kudos to https://medium.com/@pdx.lucasm/canvas-with-react-js-32e133c05258


const videoConstraints = {
  width: 512,
  height: 512,
};

const bannedCategories = ['car', 'road', 'sidewalk', 'traffic light', 'traffic sign'];

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      imageSrc: null,
      imageOutput: null,
      seed: 42,
      facingMode: "environment",
      prompt: "award-winning photo of a leafy pedestrian mall full of people, with multiracial genderqueer joggers and bicyclists and wheelchair users talking and laughing",
      maskFraction: 0.5,
    };
    this.webcamRef = React.createRef();
  }


  async componentDidMount() {
    // const segmenter = await pipeline("image-segmentation", "Xenova/segformer-b0-finetuned-cityscapes-1024-1024", {device: "wasm"});
    await setTimeout(() => console.log("starting lol"), 1000);
    const fastpostprocess = await ort.InferenceSession.create("./static/onnx/argmax.onnx", {executionProviders: ["webgpu", "wasm"]}, );
    await setTimeout(() => console.log(fastpostprocess), 1000);

    const segfeatex = new SegformerFeatureExtractor({config: {
      "do_normalize": true,
      "do_resize": true,
      "feature_extractor_type": "SegformerFeatureExtractor",
      "image_mean": [
        0.485,
        0.456,
        0.406
      ],
      "image_std": [
        0.229,
        0.224,
        0.225
      ],
      "reduce_labels": false,
      "resample": 2,
      "size": 512
    }}); // await SegformerFeatureExtractor.from_pretrained("Xenova/segformer-b0-finetuned-cityscapes-1024-1024", {device: "wasm"});
    // const text_model = await CLIPTextModelWithProjection.from_pretrained("Xenova/clip-vit-large-patch14", {device: "wasm"});
    // const text_tokenizer = await AutoTokenizer.from_pretrained("Xenova/clip-vit-large-patch14", {device: "wasm"});
    // let text_inputs = text_tokenizer("endless cars", {padding: true, truncation: true}) // text_tokenizer("a photo of a leafy pedestrian mall full of people, with multiracial genderqueer joggers and bicyclists and wheelchair users talking and laughing", {return_tensors: "tfjs"});
    // let { text_embeds } = await text_model(text_inputs)
    const mask_categories = Uint8Array.from({length: 19});
    const label2id = {
      "bicycle": 18,
      "building": 2,
      "bus": 15,
      "car": 13,
      "fence": 4,
      "motorcycle": 17,
      "person": 11,
      "pole": 5,
      "rider": 12,
      "road": 0,
      "sidewalk": 1,
      "sky": 10,
      "terrain": 9,
      "traffic light": 6,
      "traffic sign": 7,
      "train": 16,
      "truck": 14,
      "vegetation": 8,
      "wall": 3
    };
    bannedCategories.forEach((category) => {
      mask_categories[label2id[category]] = 255;
    });
    await setTimeout(() => console.log("argmax.onnx"), 1000);
    const mysegmenter = await ort.InferenceSession.create("./static/onnx/segformer.onnx", {executionProviders: ["webgpu", "wasm"]}, );
    await setTimeout(() => console.log("segformer.onnx"), 1000);
    this.setState({ mysegmenter, segfeatex, fastpostprocess, mask_categories });
    this.capture();
  }

  async capture () {
    if(this.webcamRef && this.webcamRef.current && this.webcamRef.current.getScreenshot && this.state.mysegmenter) {
      const canvas = this.webcamRef.current.getCanvas();
      if (canvas !== null) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        const imageRaw = new RawImage(imageData, canvas.width, canvas.height, 4);
        const imageResized = await imageRaw.resize(videoConstraints.width, videoConstraints.height); // TODO: resize via onnx
        // const origVisualization = new RawImage(imageResized.data, videoConstraints.width, videoConstraints.height, 4);
        // this.setState({ origVisualization });
        this.predict(imageResized);
      } else {
        console.log("image canvas is null!!!");
        requestAnimationFrame(() => this.capture());
      }
    }
    else { requestAnimationFrame(() => this.capture()); }
  }

  async predict(imageSrc) {
    const { mysegmenter, segfeatex, fastpostprocess, mask_categories } = this.state;
    const startTime = Date.now();
    const isd = Uint8ClampedArray.from(imageSrc.data);
    // const { pixel_values, pixel_mask } = await segmenter.processor([imageSrc]);
    const { pixel_values: {data: pxl } } = await segfeatex([imageSrc]);
    const {output: logits} = await mysegmenter.run( { pixel_values: new ort.Tensor("float32", pxl, [1,3,512,512]) } );
    const {output: maskImageFast} = await fastpostprocess.run({
      "seg1x19x128x128": new ort.Tensor("float32", logits.data, [1,19,128,128]),
      "mask_categories": new ort.Tensor("uint8", mask_categories, [19])
    });
    // console.log({mask, maskDataInBannedCategories});
    const maskFraction = maskImageFast.data.reduce((acc, val) => acc + val, 0.0) / (maskImageFast.data.length * 1 * 255);
    const maskVisualization = (new RawImage(maskImageFast.data, videoConstraints.width, videoConstraints.height, 1)).rgba();
    const mvd = maskVisualization.data;
    for(let i = 0; i < mvd.length; i += 4) {
      mvd[i + 0] = mvd[i+0] === 0 ? isd[i+0] : isd[i+0];
      mvd[i + 1] = mvd[i+1] === 0 ? isd[i+1] / 1.25 : 192;
      mvd[i + 2] = mvd[i+2] === 0 ? isd[i+2] : isd[i+2];
      mvd[i + 3] = 255;
    } // turn 255, 255, 255 into 32, 255, 64, a nice lightly-blue green
    const endTime = Date.now();
    this.setState({ maskFraction, maskVisualization, segmentationMillis: endTime - startTime});

    setTimeout( (() => requestAnimationFrame(() => this.capture())), 1);
  }

  render() {
    const { response, segmentationMillis, facingMode, maskVisualization, maskFraction, origVisualization } = this.state;
    return (
      <div className="App">
        <h1>BAN CARS</h1>
        <div id="maskFraction">The world is at least {Math.round(maskFraction * 100, 0)}% terrible,<br/>though I keep this from my children.</div>
        <div className="milliseconds">{segmentationMillis}ms  segmentation</div>
        <Webcam
          audio={false}
          ref={this.webcamRef}
          screenshotFormat="image/jpg"
          videoConstraints={{...videoConstraints, facingMode}}
          height={videoConstraints.height}
          width={videoConstraints.width}
          style={{height: "30vw", width: "30vw", "opaucity": "0.00001", "possition": "absolute", "top": "0px", "left": "0px"}}
        />
        <div id="response">{response}</div>
        <div>
          <Canvas id="mask"
                  height={videoConstraints['height']}
                  width={videoConstraints['width']}
                  draw={(ctx) => {
                    if(maskVisualization && maskVisualization.data) {
                      ctx.putImageData(new ImageData(maskVisualization.data, videoConstraints['width'], videoConstraints['height']), 0, 0);
                    }
                  }} style={{height: "30vw", width: "30vw"}} /><Canvas id="original" height={videoConstraints['height']} width={videoConstraints['width']} draw={(ctx) => {
                    if(origVisualization && origVisualization.data) {
                      ctx.putImageData(new ImageData(origVisualization.data, videoConstraints['width'], videoConstraints['height']), 0, 0);
                    }
                  }
                  } style={{height: "30vw", width: "30vw"}} />
        </div>
      </div>
    );
  }
}

export default App;
