import React, { useRef, useEffect } from 'react';
import './App.css';
import Webcam from 'react-webcam';

const videoConstraints = {width: 320, height: 320};

const allCategories = [
  "bicycle", "building", "bus", "car", "fence", "motorcycle", "person", "pole", "rider", "road", "sidewalk", "sky", "terrain", "traffic light", "traffic sign", "train", "truck", "vegetation", "wall"
];

const banCarsCategories = ['car', 'road', 'sidewalk', 'traffic light', 'traffic sign', 'person'];

const allSizes = [256, 320, 384, 448, 512, 640, 768, 896, 1024];
const allSteps = [8, 12, 16, 20, 50]

const modalEndpoint = "https://lsb--segformer-sdinpainting-model-inference.modal.run/"

const startingState = {
  imageSrc: null,
  seed: 42,
  size: 640,
  steps: 12,
  facingMode: "environment",
  prompt: "award-winning photo of a leafy pedestrian mall full of people, with multiracial genderqueer joggers and bicyclists and wheelchair users talking and laughing",
  bannedCategories: banCarsCategories,
  maskFraction: 50,
  millis: 0,
}


class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      ...startingState,
    };
    this.webcamRef = React.createRef();
  }


  async componentDidMount() {
    this.loadModels();
  }

  async loadModels() {
    setTimeout((() => this.capture()), 0);
  }

  async capture () {
    if(this.webcamRef && this.webcamRef.current && this.webcamRef.current.getScreenshot) {
      console.log(this.webcamRef.current.getCanvas());
      const screenshot = this.webcamRef.current.getScreenshot();
      if(screenshot) {
        this.useCapture(screenshot);
      } else {
        console.log("no screenshot!");
        setTimeout((requestAnimationFrame(() => this.capture()), 1000));
      }
    }
    else { requestAnimationFrame(() => this.capture()); }
  }

  async useCapture(screenshot) {
    console.log(screenshot.length);
    const params = new URLSearchParams({
      image_data_uri: screenshot,
      redaction_categories: this.state.bannedCategories.join(","),
      prompt: this.state.prompt,
      size: this.state.size,
      steps: this.state.steps,
      manual_seed: this.state.seed,
    });
    const startDate = Date.now();
    const response = await fetch(modalEndpoint + "?" + params.toString());
    const { image, redaction_percent } = await response.json();
    const endDate = Date.now();
    this.setState({imageSrc: image, rawSnapshot: screenshot, maskFraction: redaction_percent, millis: endDate - startDate});
    requestAnimationFrame(() => this.capture());
  }

  render() {
    const { rawSnapshot, imageSrc, millis, facingMode, maskFraction } = this.state;
    return (
      <div className="App">
        <h1>BAN CARS</h1>
        <div id="maskFraction">The world is at least {maskFraction}% terrible,<br/>though I keep this from my children.</div>
        <div className="milliseconds">{millis}ms</div>
        <Webcam
          audio={false}
          ref={this.webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{...videoConstraints, facingMode}}
          screenshotQuality={0.5}
          height={videoConstraints.height}
          width={videoConstraints.width}
          style={{"opacity": "0.0001", "position": "absolute", "top": "0px", "right": "0px"}}
        />
        <h2><img id="reimagination" src={imageSrc} /></h2>
        <h2><img id="rawSnapshot" src={rawSnapshot} /></h2>
      </div>
    );
  }
}

export default App;
