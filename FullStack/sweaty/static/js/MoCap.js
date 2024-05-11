// Desert 맵 로드 함수
async function loadDesert(MapUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      MapUrl,
      (gltf) => {
        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-2.5, -15, -20);
        mesh.scale.set(35, 35, 35);
        mesh.rotation.y = (Math.PI * 1) / 3;

        scene.add(mesh);
        // currentMap = mesh;
        console.log("맵을 로드했습니다.");
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

// Pyramid & Sphinx asset 로드 함수
async function loadPyramid(MapUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      MapUrl,
      (gltf) => {
        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-50, 0, -20);
        mesh.scale.set(0.03, 0.03, 0.03);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("asset을 로드했습니다.");
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

// Island 맵 로드 함수
async function loadIsland(MapUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      MapUrl,
      (gltf) => {
        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(1, -3.3, -1);
        mesh.scale.set(4.3, 4.3, 4.3);
        mesh.rotation.y = -(Math.PI * 1) / 4;
        scene.add(mesh);
        // currentMap = mesh;
        console.log("맵을 로드했습니다.");
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

// Island Maui asset 로드 함수
async function loadMaui(MapUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      MapUrl,
      (gltf) => {
        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-5, 0, 0);
        mesh.scale.set(1.5, 1.5, 1.5);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("asset을 로드했습니다.");
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

// Island Flower asset 로드 함수
async function loadHFlower(MapUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      MapUrl,
      (gltf) => {
        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-2, 0, 0);
        mesh.scale.set(15, 15, 15);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("flower asset을 로드했습니다.");
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

// Mountain 맵 로드 함수
async function loadMountain(MapUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      MapUrl,
      (gltf) => {
        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-2.0, -47.0, 10);
        mesh.scale.set(2, 2, 2);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("맵을 로드했습니다.");
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

// City 맵 로드 함수
async function loadCity(MapUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      MapUrl,
      (gltf) => {
        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-10, -0.5, 2);
        mesh.scale.set(0.5, 0.5, 0.5);
        mesh.rotation.y = (Math.PI * 5) / 6;
        scene.add(mesh);
        // currentMap = mesh;
        console.log("맵을 로드했습니다.");
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

async function loadMap(map) {
  switch (map) {
    case "Desert":
      MapUrl = "../../static/assets/low_poly_desert/scene.gltf";
      MapUrl2 =
        "../../static/assets 2/world_low_poly/pyramid_and_the_sphinx/scene.gltf";
      modelUrl = "../../static/model/avatar-first.vrm";
      await loadDesert(MapUrl); // MapUrl이 설정된 후 loadMap 함수 호출
      await loadModel(modelUrl);
      await loadPyramid(MapUrl2);
      stopAllAudio(); // Stop other audios
      desertAudio.play(); // Play Desert audio

      $("#btn_player").click(function (e) {
        if (desertAudio.paused == true) {
          desertAudio.play(); //재생
        } else {
          desertAudio.pause(); //일시정지
        }
      });

      desertAudio.addEventListener("ended", function () {
        //끝났을 때
      });

      console.log("Desert Map load done");
      break;
    case "Island":
      MapUrl = "../../static/assets/low_poly_island/scene.gltf";
      MapUrl2 = "../../static/assets 2/Island/Man_hawaiian/scene.gltf";
      modelUrl = "../../static/model/avatar-first.vrm";
      flowerUrl = "../../static/assets 2/Island/Hawaiian_flower/scene.gltf";
      await loadIsland(MapUrl); // MapUrl이 설정된 후 loadMap 함수 호출
      await loadModel(modelUrl);
      await loadHFlower(flowerUrl);
      // await loadMaui(MapUrl2);
      stopAllAudio(); // Stop other audios
      islandAudio.play(); // Play Island audio

      //오디오 재생
      $("#btn_player").click(function (e) {
        if (islandAudio.paused == true) {
          islandAudio.play(); //재생
        } else {
          islandAudio.pause(); //일시정지
        }
      });

      islandAudio.addEventListener("ended", function () {
        //끝났을 때
      });

      console.log("Island Map load done");
      break;
    case "Mountain":
      MapUrl = "../../static/assets/low_poly_mountain/scene.gltf";
      modelUrl = "../../static/model/avatar-first.vrm";
      await loadMountain(MapUrl); // MapUrl이 설정된 후 loadMap 함수 호출
      await loadModel(modelUrl).catch((error) => {
        console.error(error);
      });
      stopAllAudio(); // Stop other audios
      mountainAudio.play(); // Play Mountain audio

      //오디오 재생
      $("#btn_player").click(function (e) {
        if (mountainAudio.paused == true) {
          mountainAudio.play(); //재생
        } else {
          cityAudio.pause(); //일시정지
        }
      });

      mountainAudio.addEventListener("ended", function () {
        //끝났을 때
      });

      console.log("Island Map load done");
      break;
    case "City":
      MapUrl = "../../static/assets/low_poly_city/scene.gltf";
      modelUrl = "../../static/model/avatar-first.vrm";
      await loadCity(MapUrl); // MapUrl이 설정된 후 loadMap 함수 호출
      await loadModel(modelUrl);
      stopAllAudio(); // Stop other audios
      cityAudio.play(); // Play Island audio

      //오디오 재생
      $("#btn_player").click(function (e) {
        if (cityAudio.paused == true) {
          cityAudio.play(); //재생
        } else {
          cityAudio.pause(); //일시정지
        }
      });

      cityAudio.addEventListener("ended", function () {
        //끝났을 때
      });

      console.log("City Map load done");
      break;
  }
}

function setCurrentMap(map) {
  CURRENT_MAP = map;
}

// Audio Function

const desertAudio = document.getElementById("desert-audio");
const islandAudio = document.getElementById("island-audio");
const mountainAudio = document.getElementById("mountain-audio");
const cityAudio = document.getElementById("city-audio");
const correctAudio = document.getElementById("correct-audio");
const wrongAudio = document.getElementById("wrong-audio");

// Stop all audio before starting a new one
function stopAllAudio() {
  desertAudio.pause();
  islandAudio.pause();
  mountainAudio.pause();
  cityAudio.pause();
  desertAudio.currentTime = 0;
  islandAudio.currentTime = 0;
  mountainAudio.currentTime = 0;
  cityAudio.currentTime = 0;
}

// VRM 모델 로드 함수
async function loadModel(modelUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.crossOrigin = "anonymous";
    loader.load(
      modelUrl,
      (gltf) => {
        THREE.VRMUtils.removeUnnecessaryJoints(gltf.scene);

        THREE.VRM.from(gltf).then((vrm) => {
          if (currentVrm) {
            scene.remove(currentVrm.scene);
            currentVrm.dispose();
          }

          scene.add(vrm.scene);
          currentVrm = vrm;
          currentVrm.scene.rotation.y = Math.PI; // 모델을 카메라에 맞게 회전
          console.log("모델을 로드했습니다.");
          resolve(gltf);
        });
      },
      //   (progress) =>
      //     console.log(
      //       "모델 로딩 중...",
      //       100.0 * (progress.loaded / progress.total),
      //       "%"
      //     ),
      null,
      reject
    );
  });
}

async function loadTrainer1(modelUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      modelUrl,
      (gltf) => {
        const mesh = gltf.scene;
        mesh.position.set(-1.0, 1.3, -1.0);
        mesh.scale.set(0.012, 0.012, 0.012);
        scene.add(mesh);

        mixer2 = new THREE.AnimationMixer(mesh);
        const clips = gltf.animations;
        const clip1 = THREE.AnimationClip.findByName(clips, "AngryPoint");
        const clip2 = THREE.AnimationClip.findByName(clips, "HappyIdle");
        const clip3 = THREE.AnimationClip.findByName(clips, "StandingThumbsUp");

        trainer_action1 = mixer2.clipAction(clip1);
        trainer_action2 = mixer2.clipAction(clip2);
        trainer_action3 = mixer2.clipAction(clip3);

        trainer_action1.setDuration(0.005);
        trainer_action2.setDuration(0.005);
        trainer_action3.setDuration(0.005);

        // action1.play();
        trainer_action2.play();
        // trainer_action3.play();
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

async function loadTrainer2(modelUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      modelUrl,
      (gltf) => {
        console.log("hi trainer2");
        console.log(gltf);
        const mesh = gltf.scene;
        mesh.position.set(1.0, 1.0, -1.0);
        mesh.scale.set(0.012, 0.012, 0.012);
        scene.add(mesh);

        mixer3 = new THREE.AnimationMixer(mesh);
        const clips = gltf.animations;
        const clip1 = THREE.AnimationClip.findByName(clips, "AirSquat");

        trainer2_action1 = mixer3.clipAction(clip1);

        // Set the time of the animation to correspond to the 28th frame
        trainer2_action1.time = (clip1.duration / 56) * 28;

        console.log(clip1.duration);

        trainer2_action1.paused = true;
        trainer2_action1.play();
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

async function loadTexture(modelUrl) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.GLTFLoader();
    loader.load(
      modelUrl,
      (gltf) => {
        const mesh = gltf.scene;
        mesh.position.set(0, 0, 0);
        mesh.scale.set(5, 5, 5);
        scene.add(mesh);
        resolve(gltf);
      },
      null,
      reject
    );
  });
}

async function loadRGBETexture(modelUrl) {
  return new Promise((resolve, reject) => {
    const rgbeloader = new THREE.RGBELoader();
    rgbeloader.load(
      modelUrl,
      (texture) => {
        texture.mapping = THREE.EquirectangularReflectionMapping;
        scene.background = texture;
        resolve(texture);
      },
      null,
      reject
    );
  });
}

async function loadFont(modelUrl, completion) {
  return new Promise((resolve, reject) => {
    const fontLoader = new THREE.FontLoader();
    fontLoader.load(
      modelUrl,
      (font) => {
        completion(font);
        resolve(font);
      },
      null,
      reject
    );
  });
}

// ---------------------------------------------------
// 카메라 무빙
// ---------------------------------------------------

function moveCameraToTarget(targetPosition, targetLookAt, time) {
  // 현재 카메라의 위치와 방향
  const currentPosition = orbitCamera.position.clone();
  const currentLookAt = orbitControls.target.clone();

  // 이전에 실행 중이던 Tween이 있다면 중지
  if (currentTween) {
    currentTween.stop();
  }

  currentTween = new TWEEN.Tween({
    x: currentPosition.x,
    y: currentPosition.y,
    z: currentPosition.z,
    lookAtX: currentLookAt.x,
    lookAtY: currentLookAt.y,
    lookAtZ: currentLookAt.z,
  })
    .to(
      {
        x: targetPosition.x,
        y: targetPosition.y,
        z: targetPosition.z,
        lookAtX: targetLookAt.x,
        lookAtY: targetLookAt.y,
        lookAtZ: targetLookAt.z,
      },
      time * 1000
    )
    .easing(TWEEN.Easing.Quadratic.InOut)
    .onUpdate((object) => {
      // 카메라의 위치와 방향을 업데이트
      orbitCamera.position.set(object.x, object.y, object.z);
      orbitControls.target.set(object.lookAtX, object.lookAtY, object.lookAtZ);
      orbitControls.update();
    })
    .start();
}

function moveCameraCircle() {
  // 중심점 설정
  const centerPoint = new THREE.Vector3(0, 1.2, 0);

  // 현재 카메라의 위치와 방향
  const currentPosition = orbitCamera.position.clone();
  const currentLookAt = orbitControls.target.clone();

  // 이전에 실행 중이던 Tween이 있다면 중지
  if (currentTween) {
    currentTween.stop();
  }

  currentTween = new TWEEN.Tween({
    t: (Math.PI * 1) / 2,
  })
    .to(
      {
        t: Math.PI * 2 + (Math.PI * 1) / 2,
      },
      3 * 1000
    )
    .easing(TWEEN.Easing.Linear.None)
    .onUpdate((object) => {
      // 현재 각도 계산
      const angle = object.t;

      // 카메라의 위치 설정
      const newX = Math.cos(angle) * 7.0;
      const newZ = Math.sin(angle) * 7.0;
      orbitCamera.position.set(newX, 2.0, newZ);

      // 카메라가 중심을 바라보도록 방향 설정
      orbitCamera.lookAt(centerPoint);
      orbitControls.target.copy(centerPoint);
      orbitControls.update();
    })
    .onComplete(() => {
      // moveCameraDefault();
    });

  // Tween 시작
  currentTween.start();
}

function moveCameraDefault() {
  // 현재 카메라의 위치와 방향
  const currentPosition = orbitCamera.position.clone();
  const currentLookAt = orbitControls.target.clone();

  // 이전에 실행 중이던 Tween이 있다면 중지
  if (currentTween) {
    currentTween.stop();
  }

  currentTween = new TWEEN.Tween({
    x: 0.0,
    y: 2.0,
    z: 7.4,
    lookAtX: 0,
    lookAtY: 1.2,
    lookAtZ: 0,
  })
    .to(
      {
        x: 3.0,
        y: 2.0,
        z: 7.4,
        lookAtX: 0,
        lookAtY: 1.2,
        lookAtZ: 0,
      },
      3 * 1000
    )
    .easing(TWEEN.Easing.Quadratic.InOut)
    .onUpdate((object) => {
      // 카메라의 위치와 방향을 업데이트
      orbitCamera.position.set(object.x, object.y, object.z);
      orbitControls.target.set(object.lookAtX, object.lookAtY, object.lookAtZ);
      orbitControls.update();
    })
    .onComplete(() => {
      tween1.start();
    });

  const tween1 = new TWEEN.Tween({
    x: 3.0,
    y: 2.0,
    z: 7.4,
    lookAtX: 0,
    lookAtY: 1.2,
    lookAtZ: 0,
  })
    .to(
      {
        x: -3.0,
        y: 2.0,
        z: 7.4,
        lookAtX: 0,
        lookAtY: 1.2,
        lookAtZ: 0,
      },
      6 * 1000
    )
    .easing(TWEEN.Easing.Quadratic.InOut)
    .onUpdate((object) => {
      // 카메라의 위치와 방향을 업데이트
      orbitCamera.position.set(object.x, object.y, object.z);
      orbitControls.target.set(object.lookAtX, object.lookAtY, object.lookAtZ);
      orbitControls.update();
    })
    .onComplete(() => {
      tween2.start();
    });

  const tween2 = new TWEEN.Tween({
    x: -3.0,
    y: 2.0,
    z: 7.4,
    lookAtX: 0,
    lookAtY: 1.2,
    lookAtZ: 0,
  })
    .to(
      {
        x: 0.0,
        y: 2.0,
        z: 7.4,
        lookAtX: 0,
        lookAtY: 1.2,
        lookAtZ: 0,
      },
      3 * 1000
    )
    .easing(TWEEN.Easing.Quadratic.InOut)
    .onUpdate((object) => {
      // 카메라의 위치와 방향을 업데이트
      orbitCamera.position.set(object.x, object.y, object.z);
      orbitControls.target.set(object.lookAtX, object.lookAtY, object.lookAtZ);
      orbitControls.update();
    })
    .onComplete(() => {
      currentTween.start();
    });

  const tween3 = new TWEEN.Tween({
    x: currentPosition.x,
    y: currentPosition.y,
    z: currentPosition.z,
    lookAtX: currentLookAt.x,
    lookAtY: currentLookAt.y,
    lookAtZ: currentLookAt.z,
  })
    .to(
      {
        x: 0.0,
        y: 2.0,
        z: 7.4,
        lookAtX: 0,
        lookAtY: 1.2,
        lookAtZ: 0,
      },
      1 * 1000
    )
    .easing(TWEEN.Easing.Quadratic.InOut)
    .onUpdate((object) => {
      // 카메라의 위치와 방향을 업데이트
      orbitCamera.position.set(object.x, object.y, object.z);
      orbitControls.target.set(object.lookAtX, object.lookAtY, object.lookAtZ);
      orbitControls.update();
    });

  tween3.start();

  setTimeout(() => {
    currentTween.start();
  }, 1000);
}

// ---------------------------------------------------
// Animate Helper function
// ---------------------------------------------------

// 앉을때 양반다리 모양으로 접히는것이 rigRotation함수를 적용한 관절이 잘못된 것이 아닐까?
const rigRotation = (
  name,
  rotation = { x: 0, y: 0, z: 0 },
  dampener = 1,
  lerpAmount = 0.5
) => {
  if (!currentVrm) {
    return;
  }
  const Part = currentVrm.humanoid.getBoneNode(
    THREE.VRMSchema.HumanoidBoneName[name]
  );
  if (!Part) {
    return;
  }

  let euler = new THREE.Euler(
    rotation.x * dampener,
    rotation.y * dampener,
    rotation.z * dampener,
    rotation.rotationOrder || "XYZ"
  );
  let quaternion = new THREE.Quaternion().setFromEuler(euler);
  Part.quaternion.slerp(quaternion, lerpAmount); // interpolate
};

// Animate Position Helper Function
const rigPosition = (
  name,
  position = { x: 0, y: 0, z: 0 },
  dampener = 1,
  lerpAmount = 1
) => {
  if (!currentVrm) {
    return;
  }
  const Part = currentVrm.humanoid.getBoneNode(
    THREE.VRMSchema.HumanoidBoneName[name]
  );
  if (!Part) {
    return;
  }
  let vector = new THREE.Vector3(
    position.x * dampener,
    position.y * dampener,
    position.z * dampener
  );
  Part.position.lerp(vector, lerpAmount); // interpolate
};

let oldLookTarget = new THREE.Euler();
const rigFace = (riggedFace) => {
  if (!currentVrm) {
    return;
  }
  rigRotation("Neck", riggedFace.head);

  // Blendshapes and Preset Name Schema
  const Blendshape = currentVrm.blendShapeProxy;
  const PresetName = THREE.VRMSchema.BlendShapePresetName;

  // Simple example without winking. Interpolate based on old blendshape, then stabilize blink with `Kalidokit` helper function.
  // for VRM, 1 is closed, 0 is open.
  riggedFace.eye.l = lerp(
    clamp(1 - riggedFace.eye.l, 0, 1),
    Blendshape.getValue(PresetName.Blink),
    0.5
  );
  riggedFace.eye.r = lerp(
    clamp(1 - riggedFace.eye.r, 0, 1),
    Blendshape.getValue(PresetName.Blink),
    0.5
  );
  riggedFace.eye = Kalidokit.Face.stabilizeBlink(
    riggedFace.eye,
    riggedFace.head.y
  );
  Blendshape.setValue(PresetName.Blink, riggedFace.eye.l);

  // Interpolate and set mouth blendshapes
  Blendshape.setValue(
    PresetName.I,
    lerp(riggedFace.mouth.shape.I, Blendshape.getValue(PresetName.I), 0.5)
  );
  Blendshape.setValue(
    PresetName.A,
    lerp(riggedFace.mouth.shape.A, Blendshape.getValue(PresetName.A), 0.5)
  );
  Blendshape.setValue(
    PresetName.E,
    lerp(riggedFace.mouth.shape.E, Blendshape.getValue(PresetName.E), 0.5)
  );
  Blendshape.setValue(
    PresetName.O,
    lerp(riggedFace.mouth.shape.O, Blendshape.getValue(PresetName.O), 0.5)
  );
  Blendshape.setValue(
    PresetName.U,
    lerp(riggedFace.mouth.shape.U, Blendshape.getValue(PresetName.U), 0.5)
  );

  //PUPILS
  //interpolate pupil and keep a copy of the value
  let lookTarget = new THREE.Euler(
    lerp(oldLookTarget.x, riggedFace.pupil.y, 0.4),
    lerp(oldLookTarget.y, riggedFace.pupil.x, 0.4),
    0,
    "XYZ"
  );
  oldLookTarget.copy(lookTarget);
  currentVrm.lookAt.applyer.lookAt(lookTarget);
};

// ---------------------------------------------------
// startProcess
// ---------------------------------------------------

const startProcess = async () => {
  const remap = Kalidokit.Utils.remap;
  const clamp = Kalidokit.Utils.clamp;
  const lerp = Kalidokit.Vector.lerp;

  // renderer
  const renderer = new THREE.WebGLRenderer({ alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  // camera
  orbitCamera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    2000
  );

  orbitCamera.position.set(0.0, 20.0, 54.4);

  // 시작 위치와 최종 위치 정의
  const start = {
    x: orbitCamera.position.x,
    y: orbitCamera.position.y,
    z: orbitCamera.position.z,
  };
  const target = { x: 0.0, y: 2.0, z: 7.4 }; // 원하는 새로운 위치

  // controls
  orbitControls = new THREE.OrbitControls(orbitCamera, renderer.domElement);
  orbitControls.screenSpacePanning = true;
  orbitControls.target.set(0.0, 1.2, 0.0);
  orbitControls.update();

  // 맵 로드
  await loadMap(CURRENT_MAP);

  await loadTexture(
    "../static/assets/textures/star_wars_-_low_poly_hoth_skybox/scene.gltf"
  );
  console.log("skybox load done");

  await loadRGBETexture(
    "../../static/assets/kloofendal_48d_partly_cloudy_puresky_4k.hdr"
  );
  console.log("loadRGBETexture done");

  await loadFont("../../static/fonts/DungGeunMo_Regular.json", (font) => {
    const textGeometry = new THREE.TextGeometry("Count: 0", {
      font: font,
      size: 1,
      height: 0.1,
      bevelEnabled: true, // 윤곽선 활성화
      bevelSize: 0.05, // 윤곽선 크기
      bevelThickness: 0.05, // 윤곽선 두께
    });

    const textMaterial = new THREE.MeshStandardMaterial({ color: 0xffd400 }); // 텍스트 색상
    const textMesh = new THREE.Mesh(textGeometry, textMaterial);

    // 윤곽선 색상을 검정색(0x000000)으로 설정
    textMaterial.emissive.setHex(0x000000);
    textMesh.name = "countText";
    textMesh.position.set(-3, 3, 0);
    scene.add(textMesh);
  });
  console.log("loadFont done");

  // trainer1, 2 로드
  // 5/7 기준 아래 두 부분을 제외하면 ios 잘 작동함

  //   await loadTrainer1("../static/assets/trainer.glb");
  //   console.log("trainer1 load done");

  //   await loadTrainer2("../static/assets/trainer2.glb");
  //   console.log("trainer2 load done");

  console.log("loaded all model");
  document.getElementById("loading-screen").style.display = "none";

  // Main Render Loop
  const clock = new THREE.Clock();

  let mixer;
  let mixer2;
  let mixer3;
  function animate() {
    requestAnimationFrame(animate);

    if (currentVrm) {
      // Update model to render physics
      currentVrm.update(clock.getDelta());
    }
    TWEEN.update();
    orbitControls.update();
    if (mixer) {
      mixer.update(clock.getDelta());
      // console.log("믹서1")
    }

    if (mixer2) {
      mixer2.update(clock.getDelta());
      // console.log("믹서2")
    }

    if (mixer3) {
      mixer3.update(clock.getDelta());
      // console.log("믹서3")
    }

    renderer.render(scene, orbitCamera);
  }
  animate();

  // Tween.js를 사용하여 애니메이션 적용
  const tween = new TWEEN.Tween(start)
    .to(target, 3000) // 지속 시간을 밀리초 단위로 설정 (3초)
    .easing(TWEEN.Easing.Quadratic.InOut) // 이징 함수 설정 (선택 사항)
    .onUpdate(() => {
      // 애니메이션 갱신될 때마다 카메라 위치 업데이트
      orbitCamera.position.set(start.x, start.y, start.z);
    })
    .onComplete(() => {
      // 애니메이션이 끝나면 3, 2, 1 글자가 1초 간격으로 차례대로 나오게 함
      const countDownDiv = document.createElement("div");
      countDownDiv.style.position = "fixed";
      countDownDiv.style.top = "50%";
      countDownDiv.style.left = "50%";
      countDownDiv.style.transform = "translate(-50%, -50%)";
      countDownDiv.style.fontSize = "300px";
      countDownDiv.style.color = "#507A03";
      countDownDiv.style.textShadow = "0 0 10px rgba(0, 0, 0, 1)";
      countDownDiv.style.fontFamily = "DungGeunMo"; // DungGeunMo 글꼴 사용
      countDownDiv.innerText = "3";
      document.body.appendChild(countDownDiv);

      setTimeout(() => {
        countDownDiv.style.color = "#c5bb00";
        countDownDiv.innerText = "2";
      }, 1000);

      setTimeout(() => {
        countDownDiv.style.color = "#c16a00";
        countDownDiv.innerText = "1";
      }, 2000);
      setTimeout(() => {
        countDownDiv.style.color = "#c50000";
        countDownDiv.innerText = "Start!";
      }, 3000);
      setTimeout(() => {
        document.body.removeChild(countDownDiv);
      }, 4000);
    })
    .start(); // 애니메이션 시작
};

// 이전 jsonCnt를 저장할 변수 선언
let previousJsonCnt = "0";
let previousJsonAccu = "-1";
let switchAccu = 0;
console.log("초기화");

// ------------------------------
// ** ajax 모듈 실행
// ------------------------------

function getAjax(CoordinateData) {
  //보안 csrftoken
  var csrftoken = document.querySelector("meta[name=csrf_token]").content;
  startTime = performance.now();

  return $.ajax({
    url: jsonUrl, // URL을 템플릿 태그로 설정
    type: "POST",
    headers: {
      "X-CSRFToken": csrftoken, // CSRF 토큰 설정
    },
    data: CoordinateData,

    success: async function (data) {
      var jsonCnt = data.json_data0;
      var jsonAccuracy = data.json_data1;
      var processAccuracy = parseFloat(jsonAccuracy);
      var receivedDataElement2 = document.getElementById("showCnt"); // HTML 요소 선택
      receivedDataElement2.innerHTML = "Count: " + jsonCnt; // HTML
      var receivedDataElement3 = document.getElementById("showSquatAcuu"); // HTML 요소 선택
      receivedDataElement3.innerHTML =
        "SquatAccu: " + String(processAccuracy) + "%"; // HTML

      // 프로세스 바에 표시할 로직
      if (processAccuracy > 0) {
        var receivedDataElement4 = document.getElementById("text-process");
        receivedDataElement4.innerHTML = parseInt(processAccuracy);
      }

      // 현재 jsonCnt와 이전 jsonCnt를 비교하여 값이 변경되었는지 확인
      if (jsonCnt !== previousJsonCnt) {
        if (trainer2_action2 && trainer_action3) {
          trainer_action2.stop();
          trainer_action3.setDuration(0.005);
          trainer_action3.play();
        }

        setTimeout(() => {
          if (trainer_action3 && trainer_action2) {
            trainer_action3.stop();
            trainer_action2.play();
          }
        }, 3000); // 1000ms = 1초

        // 변경된 경우에만 처리 --------------------------
        previousJsonCnt = jsonCnt; // 이전 jsonCnt 업데이트

        // 이전 텍스트 메시지 제거
        const previousTextMesh = scene.getObjectByName("countText");
        scene.remove(previousTextMesh);

        // 새로운 텍스트 메시지 생성
        const fontLoader = new THREE.FontLoader();
        await loadFont("../../static/fonts/DungGeunMo_Regular.json", (font) => {
          const textGeometry = new THREE.TextGeometry("Count: " + jsonCnt, {
            font: font,
            size: 1,
            height: 0.1,
            bevelEnabled: true, // 윤곽선 활성화
            bevelSize: 0.05, // 윤곽선 크기
            bevelThickness: 0.05, // 윤곽선 두께
          });

          const textMaterial = new THREE.MeshStandardMaterial({
            color: 0x507a03,
          }); // 텍스트 색상
          const textMesh = new THREE.Mesh(textGeometry, textMaterial);

          // 윤곽선 색상을 검정색(0x000000)으로 설정
          textMaterial.emissive.setHex(0x000000);
          textMesh.name = "countText";
          textMesh.position.set(-3, 3, 0);
          scene.add(textMesh);
        });
        // test
        // 말풍선 생성 및 삭제
        const loader = new THREE.GLTFLoader();
        loader.load("../../static/assets/speech_bubble.gltf", (gltf) => {
          const mesh = gltf.scene;
          mesh.position.set(-2.6, 1.8, -0.5);
          mesh.rotation.set(0, 0.7, 0);
          mesh.scale.set(0.003, 0.003, 0.003);
          scene.add(mesh);

          // 말풍선 글씨
          fontLoader.load(
            "../../static/fonts/DungGeunMo_Regular.json",
            function (font) {
              const textGeometry = new THREE.TextGeometry("Nice!", {
                font: font,
                size: 1,
                height: 0.1,
                bevelEnabled: true, // 윤곽선 활성화
                bevelSize: 0.03, // 윤곽선 크기
                bevelThickness: 0.03, // 윤곽선 두께
              });

              const textMaterial = new THREE.MeshStandardMaterial({
                color: 0x507a03,
              }); // 텍스트 색상
              const textMesh = new THREE.Mesh(textGeometry, textMaterial);

              // 윤곽선 색상을 검정색(0x000000)으로 설정
              textMaterial.emissive.setHex(0x000000);
              textMesh.name = "cloudText";
              textMesh.position.set(-3.0, 1.6, 0.15);
              textMesh.rotation.set(0, 0.7, 0);
              textMesh.scale.set(0.5, 0.5, 0.5);
              scene.add(textMesh);
            }
          );

          // 5초 후에 GLTF를 삭제하는 함수 호출
          setTimeout(() => {
            const previousTextMesh2 = scene.getObjectByName("cloudText");
            scene.remove(previousTextMesh2); // 이전 텍스트 메시지 제거
            scene.remove(mesh); // Scene에서 mesh 제거
          }, 5000); // 1000ms = 1초
        });

        // 이전 텍스트 메시지 제거
        const previousTextMesh2 = scene.getObjectByName("cloudText");
        scene.remove(previousTextMesh2);

        // 모델의 blendShapeProxy를 가져옴
        if (currentVrm !== undefined) {
          const blendShapeProxy = currentVrm.blendShapeProxy;
          // 표정을 설정할 이름과 값 설정
          const expressionName = "fun";
          const expressionValue = 1.0; // 0부터 1 사이의 값으로 설정

          // 블렌드 쉐이프 값 설정
          blendShapeProxy.setValue(expressionName, expressionValue);

          // 3초 후에 무표정으로 돌아가는 함수 호출
          setTimeout(() => {
            // 이전 표정으로 돌아감
            blendShapeProxy.setValue(expressionName, 0);
          }, 3000); // 3초 후에 실행됨 (단위: 밀리초)
        }

        moveCameraToTarget(
          { x: 0.0, y: 1.8, z: 1.4 }, // 목표 위치
          { x: -1.0, y: 1.4, z: 0.0 }, // 목표 방향
          1 // 1초 동안 이동
        );
        setTimeout(() => {
          // moveCameraDefault();
        }, 1000);

        // 주어진 운동 횟수를 다 채웠다면, 쉬는 시간을 가짐.
        // 만약 모든 세트를 다 마치면, 다음 화면으로 넘어감
        if (previousJsonCnt == TOTAL_NUM) {
          console.log("운동 끝");
          currentSet += 1;
          // 모든 세트를 다 채운 케이스
          if (currentSet == TOTAL_SET) {
            finishExercise();
          } else {
            setBreakTime();
          }
        }
      }
      if (parseFloat(jsonAccuracy) < 80 && jsonAccuracy !== previousJsonAccu) {
        if (trainer_action2 !== undefined && trainer_action1 !== undefined) {
          trainer_action2.stop();
          trainer_action1.setDuration(0.002);
          trainer_action1.play();
        }

        setTimeout(() => {
          if (trainer_action2 !== undefined && trainer_action1 !== undefined) {
            trainer_action1.stop();
            trainer_action2.play();
          }
        }, 3000); // 1000ms = 1초

        // 모델의 blendShapeProxy를 가져옴
        if (currentVrm !== undefined) {
          const blendShapeProxy = currentVrm.blendShapeProxy;

          // 표정을 설정할 이름과 값 설정
          const expressionName = "angry";
          const expressionValue = 1.0; // 0부터 1 사이의 값으로 설정

          // 블렌드 쉐이프 값 설정
          blendShapeProxy.setValue(expressionName, expressionValue);

          // 3초 후에 무표정으로 돌아가는 함수 호출
          setTimeout(() => {
            // 이전 표정으로 돌아감
            blendShapeProxy.setValue(expressionName, 0);
          }, 3000); // 3초 후에 실행됨 (단위: 밀리초)
        }

        moveCameraToTarget(
          { x: 0.0, y: 1.8, z: 1.4 }, // 목표 위치
          { x: -1.0, y: 1.4, z: 0.0 }, // 목표 방향
          1 // 1초 동안 이동
        );
        setTimeout(() => {
          // moveCameraDefault();
        }, 1000);
      }
      // 변경된 경우에만 처리
      previousJsonAccu = jsonAccuracy; // 이전 jsonCnt 업데이트

      //aframe 가상환경 안에  텍스트 로드
      //const shoulderText2 = document.querySelector('#shoulderText2');
      //shoulderText2.setAttribute('value', `probability1: (${JSON.stringify(jsonData)})`);
    },
    error: function (xhr, status, error) {
      console.error("Error sending data:", error);
      // 오류가 발생한 경우 처리
    },
  });
}

const drawResults = (results) => {
  guideCanvas.width = videoElement.videoWidth;
  guideCanvas.height = videoElement.videoHeight;
  let canvasCtx = guideCanvas.getContext("2d");
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, guideCanvas.width, guideCanvas.height);
  // Use `Mediapipe` drawing functions
  drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
    color: "#00cff7",
    lineWidth: 4,
  });
  drawLandmarks(canvasCtx, results.poseLandmarks, {
    color: "#ff0364",
    lineWidth: 2,
  });
  drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION, {
    color: "#C0C0C070",
    lineWidth: 1,
  });
  if (results.faceLandmarks && results.faceLandmarks.length === 478) {
    //draw pupils
    drawLandmarks(
      canvasCtx,
      [results.faceLandmarks[468], results.faceLandmarks[468 + 5]],
      {
        color: "#ffe603",
        lineWidth: 2,
      }
    );
  }
  drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
    color: "#eb1064",
    lineWidth: 5,
  });
  drawLandmarks(canvasCtx, results.leftHandLandmarks, {
    color: "#00cff7",
    lineWidth: 2,
  });
  drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
    color: "#22c3e3",
    lineWidth: 5,
  });
  drawLandmarks(canvasCtx, results.rightHandLandmarks, {
    color: "#ff0364",
    lineWidth: 2,
  });
};

// Holistic 설정

const onResults = (results) => {
  if (!is_start) {
    is_start = true;
    startProcess();
  }
  // Draw landmark guides
  drawResults(results);
  // Animate model
  animateVRM(currentVrm, results);

  if (results.poseLandmarks && results.poseLandmarks.length >= 27) {
    const rightShoulderIndex = 11;
    const leftShoulderIndex = 12;
    const rightHipIndex = 23;
    const leftHipIndex = 24;
    const rightKneeIndex = 25;
    const leftKneeIndex = 26;
    const rightAnkleIndex = 27;
    const leftAnkleIndex = 28;
    const rightHeelIndex = 29;
    const leftHeelIndex = 30;
    const rightFootIndex = 31;
    const leftFootIndex = 32;

    var rightShoulderLandmark = results.poseLandmarks[rightShoulderIndex];
    var leftShoulderLandmark = results.poseLandmarks[leftShoulderIndex];
    var rightHipLandmark = results.poseLandmarks[rightHipIndex];
    var leftHipLandmark = results.poseLandmarks[leftHipIndex];
    var rightKneeLandmark = results.poseLandmarks[rightKneeIndex];
    var leftKneeLandmark = results.poseLandmarks[leftKneeIndex];
    var rightAnkleLandmark = results.poseLandmarks[rightAnkleIndex];
    var leftAnkleLandmark = results.poseLandmarks[leftAnkleIndex];
    var rightHeelLandmark = results.poseLandmarks[rightHeelIndex];
    var leftHeelLandmark = results.poseLandmarks[leftHeelIndex];
    var rightFootLandmark = results.poseLandmarks[rightFootIndex];
    var leftFootLandmark = results.poseLandmarks[leftFootIndex];

    //오른쪽 어깨
    var rightShoulderXCoordinate = rightShoulderLandmark.x;
    var rightShoulderYCoordinate = rightShoulderLandmark.y;
    var rightShoulderZCoordinate = rightShoulderLandmark.z;

    //왼쪽 어깨
    var leftShoulderXCoordinate = leftShoulderLandmark.x;
    var leftShoulderYCoordinate = leftShoulderLandmark.y;
    var leftShoulderZCoordinate = leftShoulderLandmark.z;

    //오른쪽 엉덩이
    var rightHipXCoordinate = rightHipLandmark.x;
    var rightHipYCoordinate = rightHipLandmark.y;
    var rightHipZCoordinate = rightHipLandmark.z;

    //왼쪽 엉덩이
    var leftHipXCoordinate = leftHipLandmark.x;
    var leftHipYCoordinate = leftHipLandmark.y;
    var leftHipZCoordinate = leftHipLandmark.z;

    //오른쪽 무릎
    var rightKneeXCoordinate = rightKneeLandmark.x;
    var rightKneeYCoordinate = rightKneeLandmark.y;
    var rightKneeZCoordinate = rightKneeLandmark.z;

    //왼쪽 무릎
    var leftKneeXCoordinate = leftKneeLandmark.x;
    var leftKneeYCoordinate = leftKneeLandmark.y;
    var leftKneeZCoordinate = leftKneeLandmark.z;

    //오른쪽 발목
    var rightAnkleXCoordinate = rightAnkleLandmark.x;
    var rightAnkleYCoordinate = rightAnkleLandmark.y;
    var rightAnkleZCoordinate = rightAnkleLandmark.z;

    //왼쪽 발목
    var leftAnkleXCoordinate = leftAnkleLandmark.x;
    var leftAnkleYCoordinate = leftAnkleLandmark.y;
    var leftAnkleZCoordinate = leftAnkleLandmark.z;

    //오른쪽 발 뒤꿈치
    var rightHeelXCoordinate = rightHeelLandmark.x;
    var rightHeelYCoordinate = rightHeelLandmark.y;
    var rightHeelZCoordinate = rightHeelLandmark.z;

    //왼쪽 발 뒤꿈치
    var leftHeelXCoordinate = leftHeelLandmark.x;
    var leftHeelYCoordinate = leftHeelLandmark.y;
    var leftHeelZCoordinate = leftHeelLandmark.z;

    //오른쪽 발가락
    var rightFootXCoordinate = rightFootLandmark.x;
    var rightFootYCoordinate = rightFootLandmark.y;
    var rightFootZCoordinate = rightFootLandmark.z;

    //왼쪽 발가락
    var leftFootXCoordinate = leftFootLandmark.x;
    var leftFootYCoordinate = leftFootLandmark.y;
    var leftFootZCoordinate = leftFootLandmark.z;

    var CoordinateData = {
      rightShoulderXCoordinate: parseFloat(rightShoulderXCoordinate),
      rightShoulderYCoordinate: parseFloat(rightShoulderYCoordinate),
      //"rightShoulderZCoordinate": parseFloat(rightShoulderZCoordinate),

      leftShoulderXCoordinate: parseFloat(leftShoulderXCoordinate),
      leftShoulderYCoordinate: parseFloat(leftShoulderYCoordinate),
      //"leftShoulderZCoordinate": parseFloat(leftShoulderZCoordinate),

      rightHipXCoordinate: parseFloat(rightHipXCoordinate),
      rightHipYCoordinate: parseFloat(rightHipYCoordinate),
      //"rightHipZCoordinate": parseFloat(rightHipZCoordinate),

      leftHipXCoordinate: parseFloat(leftHipXCoordinate),
      leftHipYCoordinate: parseFloat(leftHipYCoordinate),
      //"leftHipZCoordinate": parseFloat(leftHipZCoordinate),

      rightKneeXCoordinate: parseFloat(rightKneeXCoordinate),
      rightKneeYCoordinate: parseFloat(rightKneeYCoordinate),
      //"rightKneeZCoordinate": parseFloat(rightKneeZCoordinate),

      leftKneeXCoordinate: parseFloat(leftKneeXCoordinate),
      leftKneeYCoordinate: parseFloat(leftKneeYCoordinate),
      //"leftKneeZCoordinate": parseFloat(leftKneeZCoordinate),

      rightAnkleXCoordinate: parseFloat(rightAnkleXCoordinate),
      rightAnkleYCoordinate: parseFloat(rightAnkleYCoordinate),
      //"rightAnkleZCoordinate": parseFloat(rightAnkleZCoordinate),

      leftAnkleXCoordinate: parseFloat(leftAnkleXCoordinate),
      leftAnkleYCoordinate: parseFloat(leftAnkleYCoordinate),
      //"leftAnkleZCoordinate": parseFloat(leftAnkleZCoordinate),

      rightHeelXCoordinate: parseFloat(rightHeelXCoordinate),
      rightHeelYCoordinate: parseFloat(rightHeelYCoordinate),
      //"rightHeelZCoordinate": parseFloat(rightHeelZCoordinate),

      leftHeelXCoordinate: parseFloat(leftHeelXCoordinate),
      leftHeelYCoordinate: parseFloat(leftHeelYCoordinate),
      //"leftHeelZCoordinate": parseFloat(leftHeelZCoordinate),

      rightFootXCoordinate: parseFloat(rightHeelXCoordinate),
      rightFootYCoordinate: parseFloat(rightHeelYCoordinate),
      //"rightFootZCoordinate": parseFloat(rightHeelZCoordinate),

      leftFootXCoordinate: parseFloat(leftFootXCoordinate),
      leftFootYCoordinate: parseFloat(leftFootYCoordinate),
      //"leftFootZCoordinate": parseFloat(leftFootZCoordinate)
    };

    myxhr = getAjax(CoordinateData);
  }
};

// Animate VRM
/* VRM Character Animator */
const animateVRM = (vrm, results) => {
  if (!vrm) {
    return;
  }

  // Take the results from `Holistic` and animate character based on its Face, Pose, and Hand Keypoints.
  let riggedPose, riggedLeftHand, riggedRightHand, riggedFace;

  const faceLandmarks = results.faceLandmarks;
  // Pose 3D Landmarks are with respect to Hip distance in meters
  const pose3DLandmarks = results.ea;
  // Pose 2D landmarks are with respect to videoWidth and videoHeight
  const pose2DLandmarks = results.poseLandmarks;
  // Be careful, hand landmarks may be reversed
  const leftHandLandmarks = results.rightHandLandmarks;
  const rightHandLandmarks = results.leftHandLandmarks;

  // Animate Face
  if (faceLandmarks) {
    riggedFace = Kalidokit.Face.solve(faceLandmarks, {
      runtime: "mediapipe",
      video: videoElement,
    });
    rigFace(riggedFace);
  }
  // Animate Pose
  if (pose2DLandmarks && pose3DLandmarks) {
    riggedPose = Kalidokit.Pose.solve(pose3DLandmarks, pose2DLandmarks, {
      runtime: "mediapipe",
      video: videoElement,
    });

    // 엉덩이 관절 노드를 얻기
    const hipsNode = vrm.humanoid.getBoneNode(
      THREE.VRMSchema.HumanoidBoneName.Hips
    );
    // 노드의 위치 벡터를 가져오기
    const hipsPosition = new THREE.Vector3();
    if (hipsNode) {
      hipsPosition.setFromMatrixPosition(hipsNode.matrixWorld);
    }

    // 오른쪽 발 관절 노드를 얻기
    const rightFootNode = vrm.humanoid.getBoneNode(
      THREE.VRMSchema.HumanoidBoneName.RightFoot
    );
    // 노드의 위치 벡터를 저장할 변수
    const rightFootPosition = new THREE.Vector3();
    if (rightFootNode) {
      rightFootPosition.setFromMatrixPosition(rightFootNode.matrixWorld);
    }

    // 왼쪽 발 관절 노드를 얻기
    const leftFootNode = vrm.humanoid.getBoneNode(
      THREE.VRMSchema.HumanoidBoneName.LeftFoot
    );
    // 노드의 위치 벡터를 저장할 변수
    const leftFootPosition = new THREE.Vector3();
    if (leftFootNode) {
      leftFootPosition.setFromMatrixPosition(leftFootNode.matrixWorld);
    }

    // 어느쪽 발이 더 높은지 판단
    if (rightFootPosition.y > leftFootPosition.y) {
      rigPosition(
        "Hips",
        {
          x: hipsPosition.x,
          y: hipsPosition.y - leftFootPosition.y + 0.13,
          z: 0,
        },
        1,
        1
      );
    } else {
      rigPosition(
        "Hips",
        {
          x: hipsPosition.x,
          y: hipsPosition.y - rightFootPosition.y + 0.13,
          z: 0,
        },
        1,
        1
      );
    }

    // 오른쪽 팔 관절 노드를 얻기
    const rightHandNode = vrm.humanoid.getBoneNode(
      THREE.VRMSchema.HumanoidBoneName.RightHand
    );
    // 노드의 위치 벡터를 저장할 변수
    const rightHandPosition = new THREE.Vector3();
    if (rightHandNode) {
      rightHandPosition.setFromMatrixPosition(rightHandNode.matrixWorld);
    }

    // 머리 관절 노드를 얻기
    const headNode = vrm.humanoid.getBoneNode(
      THREE.VRMSchema.HumanoidBoneName.Head
    );
    // 노드의 위치 벡터를 저장할 변수
    const headPosition = new THREE.Vector3();
    if (headNode) {
      headPosition.setFromMatrixPosition(headNode.matrixWorld);
    }

    if (rightHandPosition.y - headPosition.y > 0.2) moveCameraCircle();

    // 머리의 위치에따라 trainer2.glb의 애니메이션이 따라가도록 조정
    if (trainer2_action1) {
      let cnt = 0;
      for (let i = 1.48; i > 1.0; i -= 0.02) {
        if (headPosition.y > i) {
          trainer2_action1.time = (1.8666666746139526 / 56) * cnt;
          trainer2_action1.paused = true;
          trainer2_action1.play();
          break;
        }
        cnt += 1;
      }
    }

    rigRotation("Hips", riggedPose.Hips.rotation);

    riggedPose.LeftUpperLeg.y = riggedPose.LeftUpperLeg.y * -0.2;
    riggedPose.RightUpperLeg.y = riggedPose.RightUpperLeg.y * -0.2;
    rigRotation("Chest", riggedPose.Chest);
    rigRotation("Spine", riggedPose.Spine);
    rigRotation("RightUpperArm", riggedPose.RightUpperArm);
    rigRotation("RightLowerArm", riggedPose.RightLowerArm);
    rigRotation("LeftUpperArm", riggedPose.LeftUpperArm);
    rigRotation("LeftLowerArm", riggedPose.LeftLowerArm);
    rigRotation("LeftUpperLeg", riggedPose.LeftUpperLeg);
    rigRotation("LeftLowerLeg", riggedPose.LeftLowerLeg);
    rigRotation("RightUpperLeg", riggedPose.RightUpperLeg);
    rigRotation("RightLowerLeg", riggedPose.RightLowerLeg);
  }

  // Animate Hands
  if (leftHandLandmarks) {
    riggedLeftHand = Kalidokit.Hand.solve(leftHandLandmarks, "Left");
    rigRotation("LeftHand", {
      // Combine pose rotation Z and hand rotation X Y
      z: riggedPose.LeftHand.z,
      y: riggedLeftHand.LeftWrist.y,
      x: riggedLeftHand.LeftWrist.x,
    });
    rigRotation("LeftRingProximal", riggedLeftHand.LeftRingProximal);
    rigRotation("LeftRingIntermediate", riggedLeftHand.LeftRingIntermediate);
    rigRotation("LeftRingDistal", riggedLeftHand.LeftRingDistal);
    rigRotation("LeftIndexProximal", riggedLeftHand.LeftIndexProximal);
    rigRotation("LeftIndexIntermediate", riggedLeftHand.LeftIndexIntermediate);
    rigRotation("LeftIndexDistal", riggedLeftHand.LeftIndexDistal);
    rigRotation("LeftMiddleProximal", riggedLeftHand.LeftMiddleProximal);
    rigRotation(
      "LeftMiddleIntermediate",
      riggedLeftHand.LeftMiddleIntermediate
    );
    rigRotation("LeftMiddleDistal", riggedLeftHand.LeftMiddleDistal);
    rigRotation("LeftThumbProximal", riggedLeftHand.LeftThumbProximal);
    rigRotation("LeftThumbIntermediate", riggedLeftHand.LeftThumbIntermediate);
    rigRotation("LeftThumbDistal", riggedLeftHand.LeftThumbDistal);
    rigRotation("LeftLittleProximal", riggedLeftHand.LeftLittleProximal);
    rigRotation(
      "LeftLittleIntermediate",
      riggedLeftHand.LeftLittleIntermediate
    );
    rigRotation("LeftLittleDistal", riggedLeftHand.LeftLittleDistal);
  }
  if (rightHandLandmarks) {
    riggedRightHand = Kalidokit.Hand.solve(rightHandLandmarks, "Right");
    rigRotation("RightHand", {
      // Combine Z axis from pose hand and X/Y axis from hand wrist rotation
      z: riggedPose.RightHand.z,
      y: riggedRightHand.RightWrist.y,
      x: riggedRightHand.RightWrist.x,
    });
    rigRotation("RightRingProximal", riggedRightHand.RightRingProximal);
    rigRotation("RightRingIntermediate", riggedRightHand.RightRingIntermediate);
    rigRotation("RightRingDistal", riggedRightHand.RightRingDistal);
    rigRotation("RightIndexProximal", riggedRightHand.RightIndexProximal);
    rigRotation(
      "RightIndexIntermediate",
      riggedRightHand.RightIndexIntermediate
    );
    rigRotation("RightIndexDistal", riggedRightHand.RightIndexDistal);
    rigRotation("RightMiddleProximal", riggedRightHand.RightMiddleProximal);
    rigRotation(
      "RightMiddleIntermediate",
      riggedRightHand.RightMiddleIntermediate
    );
    rigRotation("RightMiddleDistal", riggedRightHand.RightMiddleDistal);
    rigRotation("RightThumbProximal", riggedRightHand.RightThumbProximal);
    rigRotation(
      "RightThumbIntermediate",
      riggedRightHand.RightThumbIntermediate
    );
    rigRotation("RightThumbDistal", riggedRightHand.RightThumbDistal);
    rigRotation("RightLittleProximal", riggedRightHand.RightLittleProximal);
    rigRotation(
      "RightLittleIntermediate",
      riggedRightHand.RightLittleIntermediate
    );
    rigRotation("RightLittleDistal", riggedRightHand.RightLittleDistal);
  }
};

// ------------------------------------------------
// 쉬는시간, 운동 끝날 때 설정
// ------------------------------------------------

// ---------------------------------------
// ** 세트 수, 1세트당 횟수, 쉬는시간 변수 **

const TOTAL_SET = 2;
const TOTAL_NUM = 10;
const BREAK_TIME_NUM = 2; // 단위 : 초

let currentSet = 0;
// previousJsonCnt = "0"; -> 현재 카운트 횟수

// 쉬는시간 적용 함수
function setBreakTime() {
  console.log("setBreakTime");
  camera.stop();
  console.log(myxhr);
  myxhr.abort();
  const countDownDiv = document.createElement("div");
  countDownDiv.style.position = "fixed";
  countDownDiv.style.top = "50%";
  countDownDiv.style.left = "50%";
  countDownDiv.style.transform = "translate(-50%, -50%)";
  countDownDiv.style.fontSize = "300px";
  countDownDiv.style.color = "#507A03";
  countDownDiv.style.textShadow = "0 0 10px rgba(0, 0, 0, 1)";
  countDownDiv.style.fontFamily = "DungGeunMo"; // DungGeunMo 글꼴 사용
  countDownDiv.innerText = "3";
  document.body.appendChild(countDownDiv);
  for (let i = 0; i < BREAK_TIME_NUM; i++) {
    setTimeout(() => {
      countDownDiv.style.color = "#c5bb00";
      countDownDiv.innerText = (BREAK_TIME_NUM - i).toString();
    }, (i + 1) * 1000);
  }
  setTimeout(() => {
    // 쉬는시간 끝, 다시 시작
    document.body.removeChild(countDownDiv);
    camera.start();
    myxhr = getAjax(CoordinateData);
  }, (BREAK_TIME_NUM + 1) * 1000);
}

// 모든 운동 끝났을 때
function finishExercise() {
  console.log("finishExercise");
  camera.stop();
  console.log(myxhr);
  myxhr.abort();
  const countDownDiv = document.createElement("div");
  countDownDiv.style.position = "fixed";
  countDownDiv.style.top = "50%";
  countDownDiv.style.left = "50%";
  countDownDiv.style.transform = "translate(-50%, -50%)";
  countDownDiv.style.fontSize = "300px";
  countDownDiv.style.color = "#507A03";
  countDownDiv.style.textShadow = "0 0 10px rgba(0, 0, 0, 1)";
  countDownDiv.style.fontFamily = "DungGeunMo"; // DungGeunMo 글꼴 사용
  countDownDiv.innerText = "Great!";
  document.body.appendChild(countDownDiv);
}

// ------------------------
// 전역 변수
// ------------------------

var CURRENT_MAP;

let orbitCamera;
let orbitControls;
let scene;
let currentTween;

let trainer_action1;
let trainer_action2;
let trainer_action3;
let trainer2_action1;
let trainer2_action2;

/* THREEJS WORLD SETUP */
let currentVrm = undefined;

/* SETUP MEDIAPIPE HOLISTIC INSTANCE */
const holistic = new Holistic({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1635989137/${file}`;
  },
});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
  refineFaceLandmarks: false,
});
// Pass holistic a callback function
holistic.onResults(onResults);

let videoElement = document.querySelector(".input_video"),
  guideCanvas = document.querySelector("canvas.guides");
let frameCounter = 0;

var is_start = false;
// Use `Mediapipe` utils to get camera - lower resolution = higher fps

const setupDefaultSettings = () => {
  // scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x84ffff);

  // light
  const light = new THREE.DirectionalLight(0xffffff);
  light.position.set(0.0, 20.0, 54.4); // 광원의 실제 위치를 변경하여 머리 위로 이동
  // light.position.set(0.0, 0.0, 0.4); // 광원의 실제 위치를 변경하여 머리 위로 이동

  light.intensity = 1.5; // 밝기 조절
  scene.add(light);
};

setupDefaultSettings();

const setupCamera = () => {
  const camera = new Camera(videoElement, {
    onFrame: async () => {
      if (holistic !== undefined) {
        await holistic.send({ image: videoElement });
      }

      frameCounter++;
    },
    width: 640,
    height: 480,
    // facingMode: "environment", //학교에서 빌린 웹캠은 후면카메라로 인식되어서 코드 추가함
  });
  camera.start();
};
