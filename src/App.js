import React, { useRef, useEffect } from 'react';
import './App.css';
import Webcam from 'react-webcam';

const videoConstraints = {width: 320, height: 320};

const allCategories = [
  "bicycle", "building", "bus", "car", "fence", "motorcycle", "person", "pole", "rider", "road", "sidewalk", "sky", "terrain", "traffic light", "traffic sign", "train", "truck", "vegetation", "wall"
];

const banCarsCategories = ['car', 'road', 'sidewalk', 'traffic light', 'traffic sign', 'person'];

const allSizes = [256, 320, 384, 448, 512, 640, 768, 896, 1024, 2048];
const allSteps = [8, 12, 16, 20, 50]
const allConditionings = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0];

const presets = ['urban', 'desert', 'kittens', 'custom'];

const modalEndpoint = "https://lsb--segformer-sdinpainting-model-inference.modal.run/"

const urbanPrompt = "pen and ink watercolor of a leafy pedestrian mall full of people, with multiracial genderqueer joggers and bicyclists and wheelchair users talking and laughing";
const desertPrompt = "beautiful desert after a rain at midnight, large boulders and pools of water and a dusting of snow";
const kittensPrompt = "kittens poking their heads out from blankets";

const startingState = {
  imageSrc: null,
  seed: 42,
  size: 640,
  steps: 94111,
  facingMode: "environment",
  prompt: urbanPrompt,
  bannedCategories: banCarsCategories,
  maskFraction: 50,
  millis: 0,
  seg_time: 0,
  paint_time: 0,
  conditioning: 0.9,
  configureExpanded: false,
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
    const params = new URLSearchParams({
      image_data_uri: screenshot,
      redaction_categories: this.state.bannedCategories.join(","),
      prompt: this.state.prompt,
      size: this.state.size,
      steps: this.state.steps,
      manual_seed: this.state.seed,
      controlnet_conditioning: this.state.conditioning,
    });
    const startDate = Date.now();
    const response = await fetch(modalEndpoint + "?" + params.toString());
    const { image, redaction_percent, seg_time, paint_time, compiled } = await response.json();
    console.log(compiled);
    const endDate = Date.now();
    this.setState({imageSrc: image, rawSnapshot: screenshot, maskFraction: redaction_percent, millis: endDate - startDate, seg_time, paint_time});
    requestAnimationFrame(() => this.capture());
  }

  render() {
    const { rawSnapshot, imageSrc, size, steps, prompt, millis, facingMode, maskFraction, configureExpanded, seg_time, paint_time } = this.state;
    return (
      <div className="App">
      <h1>BAN CARS</h1>
      <div id="maskFraction">The world is at least {maskFraction}% terrible,<br/>though I keep this from my children.</div>
      <div className="milliseconds">({Math.round(seg_time * 1000)}/{Math.round(paint_time * 1000)}) {millis}ms</div>
      <Webcam
        audio={false}
        ref={this.webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={{...videoConstraints, facingMode}}
        screenshotQuality={0.5}
        height={videoConstraints.height}
        width={videoConstraints.width}
        style={{"opacity": "0.0001", "position": "fixed", "top": "0px", "right": "0px"}}
      />
      <div><img id="reimagination" src={imageSrc} /><img id="rawSnapshot" src={rawSnapshot} /></div>
      <div>
        <label>Preset:&nbsp;</label>
        <select
          value={(prompt === urbanPrompt) ? 'urban' : (prompt === desertPrompt ? 'desert' : (prompt === kittensPrompt ? 'kittens' : 'custom'))}
          onChange={(e) => {
            const selectedPreset = e.target.value;
            let newPrompt;
            let newConfigureExpanded;
            switch (selectedPreset) {
              case 'urban':
                newPrompt = urbanPrompt;
                newConfigureExpanded = false;
                break;
              case 'desert':
                newPrompt = desertPrompt;
                newConfigureExpanded = false;
                break;
              case 'kittens':
                newPrompt = kittensPrompt;
                newConfigureExpanded = false;
                break;
              case 'custom':
                newPrompt = this.state.prompt + ' ';
                newConfigureExpanded = true;
                break;
              default:
                newPrompt = this.state.prompt;
            }
            this.setState({ prompt: newPrompt, configureExpanded: newConfigureExpanded });
          }}
        >
          {presets.map((preset) => (
            <option key={preset} value={preset}>{preset}</option>
          ))}
        </select>
      </div>
      {/* <button onClick={(e) => this.setState({configureExpanded: !configureExpanded})}>⚙️{configureExpanded ? "🔼" : "🔽"}</button> */}
      <div id="configure" style={{display: configureExpanded ? "block" : "none", width: "95%", margin: "auto"}}>
        <div>
        <label>Prompt:</label><br/>
        <textarea rows={5} style={{display: "block", width: "95%"}} value={prompt} onChange={(e) => this.setState({prompt: e.target.value})} />
        </div>
        <div>
        <label>Banned Categories:</label>
        <div>
          {allCategories.map((category) => (
          <span key={category}>
            <input
            type="checkbox"
            id={category}
            value={category}
            checked={this.state.bannedCategories.includes(category)}
            onChange={(e) => {
              const newBannedCategories = e.target.checked
              ? [...this.state.bannedCategories, category]
              : this.state.bannedCategories.filter((c) => c !== category);
              this.setState({ bannedCategories: newBannedCategories });
            }}
            />
            <label htmlFor={category}>{category}</label>
            &nbsp;
          </span>
          ))}
        </div>
        </div>
        {/* <div>
        <label>Steps: {steps}</label>
        <input type="range" min={0} max={allSteps.length - 1} step="1" value={allSteps.indexOf(steps)} onChange={(e) => this.setState({steps: allSteps[parseInt(e.target.value)]})} />
        </div> */}
        <div>
        <label>Size: {size}</label>
        <input type="range" min={0} max={allSizes.length - 1} step="1" value={allSizes.indexOf(size)} onChange={(e) => this.setState({size: allSizes[parseInt(e.target.value)]})} />
        </div>
        <div>
        <label>Conditioning: <code>{this.state.conditioning.toFixed(1)}</code></label>
        <input type="range" min={0} max={allConditionings.length - 1} step="1" value={allConditionings.indexOf(this.state.conditioning)} onChange={(e) => this.setState({conditioning: allConditionings[parseInt(e.target.value)]})} />
        </div>
        <div>
        <label>Facing Mode:</label>
        <select value={this.state.facingMode} onChange={(e) => this.setState({facingMode: e.target.value})}>
          <option value="user">User</option>
          <option value="environment">Environment</option>
        </select>
        </div>
        <div>
        <label>Seed:</label>
        <input type="number" value={this.state.seed} onChange={(e) => this.setState({seed: e.target.value})} />
        </div>
      </div>
      </div>
    );
  }
}

export default App;
