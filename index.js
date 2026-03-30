const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    icon: path.join(__dirname, 'assets/icon.png'),
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    title: "Wi-Fi CSI Human Sensing Pipeline"
  });

  mainWindow.loadFile('loading.html');

  const targetUrl = 'http://127.0.0.1:8050/';
  let loaded = false;

  const tryLoad = () => {
    if (!mainWindow || loaded) return;
    fetch(targetUrl).then(response => {
      if(response.ok) {
        mainWindow.loadURL(targetUrl);
        loaded = true;
      } else {
        setTimeout(tryLoad, 2000);
      }
    }).catch((err) => {
      console.log('Backend not ready, retrying...', err.message);
      setTimeout(tryLoad, 2000);
    });
  };

  setTimeout(tryLoad, 3000); // Start checking after 3s

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

function startPythonBackend() {
  const scriptPath = path.join(__dirname, 'src', 'app.py');
  
  // Decide whether to use replay mode if data exists
  let args = [scriptPath];
  if (fs.existsSync(path.join(__dirname, 'models', 'synthetic_csi.csv'))) {
      args.push('--replay');
      args.push(path.join(__dirname, 'models', 'synthetic_csi.csv'));
  }
  
  pythonProcess = spawn('python', args);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });
}

app.whenReady().then(() => {
  startPythonBackend();
  createWindow();
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

app.on('activate', function () {
  if (mainWindow === null) {
    createWindow();
  }
});