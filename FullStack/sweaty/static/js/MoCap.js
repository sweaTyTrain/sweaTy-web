
document.addEventListener("DOMContentLoaded", (_) => {
    let canvas = document.createElement("canvas");
    let c = canvas.getContext("2d");
  
    document.body.appendChild(canvas);
  
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  
    const config = {
      size: 3,
      number: 20,
      fill: 0.1
    };
  
    const colorScheme = [
      "#f2f2f2",
      "#fa709a",
      "#E7DE2F",
      "#fee140",
      "#FFAEAE",
      "#77A5D8",
      "#c2e9fb",
      "#96e6a1",
      "#453a94"
    ];
  
    /** Begin Firework -> 폭죽의 속도 조절 **/
    function Firework() {
      this.radius = Math.random() * config.size + 1;
      this.x = canvas.width / 2;
      this.y = canvas.height + this.radius;
      this.color = colorScheme[Math.floor(Math.random() * colorScheme.length)];
      this.velocity = {
        x: Math.random() * 8 - 4,
        y: Math.random() * 4 + 4
      };
      this.maxY = (Math.random() * canvas.height) / 4 + canvas.height / 10;
      this.life = false;
    }
  
    Firework.prototype.draw = function (c) {
      c.beginPath();
      c.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      c.fillStyle = this.color;
      c.fill();
      c.closePath();
    };
  
    Firework.prototype.maximumY = function () {
      if (this.y <= this.maxY || this.x <= 0 || this.x >= canvas.width) {
        this.life = true;
        for (let i = 0; i < 10; i++) {
          sparkArray.push(new Spark(this.x, this.y, this.radius, this.color));
        }
      }
    };
  
    Firework.prototype.update = function (c) {
      this.y -= this.velocity.y;
      this.x += this.velocity.x;
  
      this.maximumY();
  
      this.draw(c);
    };
    /** End Firework**/
    /** Spark **/
    function Spark(x, y, radius, color) {
      this.x = x;
      this.y = y;
      this.radius = radius / 2;
      this.color = color;
      this.velocity = {
        x: Math.random() * 3 - 1,
        y: Math.random() * 3 - 1
      };
      this.curve = 0.025;
      this.life = 140;
    }
  
    Spark.prototype.draw = function (c) {
      c.beginPath();
      c.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      c.fillStyle = this.color;
      c.fill();
      c.closePath();
    };
  
    Spark.prototype.update = function (c) {
      this.velocity.y -= this.curve;
      this.life -= 1;
      this.x += this.velocity.x;
      this.y -= this.velocity.y;
      this.draw(c);
    };
    /** End Spark **/
  
    let fireworkArray = [];
    let sparkArray = [];
  
    function init() {
      if (fireworkArray.length < config.number) {
        fireworkArray.push(new Firework());
      }
    }
  
    function animate() {
        let startTime = Date.now(); // 시작 시간 기록
    
        function frame() {
            let currentTime = Date.now(); // 현재 시간 기록
            let elapsedTime = (currentTime - startTime) / 1000; // 경과 시간 (초 단위)
            // 오디오 요소 생성
            var audio = new Audio('FullStack/sweaty/avatar/motion_effect/mp3-output-ttsfree(dot)com (1).mp3');

            // 재생
            audio.play();
    
            if (elapsedTime < 5) { // 5초가 지날 때까지 애니메이션 실행

                // "운동 정확도가 90% 넘었어요!" 텍스트 추가
                let textElement = document.createElement("div");
                textElement.textContent = "운동 정확도가 90% 넘었어요 !";
                textElement.style.position = "absolute";
                textElement.style.top = "50%";
                textElement.style.left = "50%";
                textElement.style.transform = "translate(-50%, -50%)";
                textElement.style.color = "white";
                textElement.style.fontSize = "35px"; // 텍스트 크기 지정
                textElement.style.width = "300px"; // 텍스트 필드 너비 지정
                textElement.style.textAlign = "center"; // 가운데 정렬
                textElement.style.whiteSpace = "nowrap"; // 한 줄로 출력
                textElement.style.fontWeight = "bold"; // 볼드체로 출력


                document.body.appendChild(textElement);

                window.requestAnimationFrame(frame);
    
                c.fillStyle = `rgba(0,0,0,${config.fill})`;
                c.fillRect(0, 0, canvas.width, canvas.height);
    
                fireworkArray.forEach((fw, index) => {
                    fw.update(c);
    
                    if (fw.life) {
                        fireworkArray.splice(index, 1);
                    }
                });
    
                sparkArray.forEach((s, index) => {
                    if (s.life <= 0) {
                        sparkArray.splice(index, 1);
                    }
    
                    s.update(c);
                });
    
                init();
            } else { // 애니메이션이 종료되면

                // 모든 컴포넌트를 사라지게 함
                fireworkArray = [];
                sparkArray = [];
                document.body.removeChild(canvas);
            }
        }
    
        frame(); // frame 함수 호출
    }           
  
    animate();
  });
  



  












const remap = Kalidokit.Utils.remap;
const clamp = Kalidokit.Utils.clamp;
const lerp = Kalidokit.Vector.lerp;

/* THREEJS WORLD SETUP */
let currentVrm;

// renderer
const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// camera
const orbitCamera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 1000);
orbitCamera.position.set(0.0, 0.0, 4.4);

// controls
const orbitControls = new THREE.OrbitControls(orbitCamera, renderer.domElement);
orbitControls.screenSpacePanning = true;
orbitControls.target.set(0.0, 1.2, 0.0);
orbitControls.update();

// scene
const scene = new THREE.Scene();

// light
const light = new THREE.DirectionalLight(0xffffff);
light.position.set(1.0, 1.0, 1.0).normalize();
scene.add(light);

const textureLoader = new THREE.TextureLoader();
const tilesBaseColor = textureLoader.load("../../static/img/Metal_Tiles_003_basecolor.jpg");
const tilesNormalMap = textureLoader.load("../../static/img/Metal_Tiles_003_normal.jpg");
const tilesHeightMap = textureLoader.load("../../static/img/Metal_Tiles_003_height.png");
const tilesRoughnessMap = textureLoader.load("../../static/img/Metal_Tiles_003_roughness.jpg");
const tilesAmbientOcclusionMap = textureLoader.load("../../static/img/Metal_Tiles_003_ambientOcclusion.jpg");
const tilesMetallicMap = textureLoader.load("../../static/img/Metal_Tiles_003_metallic.jpg");

const plane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2, 512, 512), 
    new THREE.MeshStandardMaterial(
        {
            map: tilesBaseColor,
            normalMap: tilesNormalMap,
            displacementMap: tilesHeightMap,
            displacementScale: 0.1,
            roughnessMap: tilesRoughnessMap,
            roughness: 0.5,
            aoMap: tilesAmbientOcclusionMap,
        }

))

plane.rotation.x = Math.PI * 3/2;
plane.geometry.attributes.uv2 = plane.geometry.attributes.uv;
scene.add(plane);

// // 바닥 생성
// const groundSize = 10; // 바닥의 크기
// const groundMaterial = new THREE.MeshBasicMaterial({ color: 0x808080, side: THREE.DoubleSide }); // 바닥의 재질
// const groundGeometry = new THREE.PlaneGeometry(groundSize, groundSize); // 바닥의 geometry 생성
// const groundMesh = new THREE.Mesh(groundGeometry, groundMaterial);// 바닥의 mesh 생성
// groundMesh.rotation.x = Math.PI / 2; // 바닥의 회전 설정 (예: 바닥을 x축 주변으로 90도 회전)
// scene.add(groundMesh);  // 씬에 바닥 추가

// Main Render Loop
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);

    if (currentVrm) {
        // Update model to render physics
        currentVrm.update(clock.getDelta());
    }
    renderer.render(scene, orbitCamera);
}
animate();

/* VRM CHARACTER SETUP */

// Import Character VRM
const loader = new THREE.GLTFLoader();
loader.crossOrigin = "anonymous";
// Import model from URL, add your own model here
loader.load(
    modelUrl,

    (gltf) => {
        THREE.VRMUtils.removeUnnecessaryJoints(gltf.scene);

        THREE.VRM.from(gltf).then((vrm) => {
            scene.add(vrm.scene);
            currentVrm = vrm;
            currentVrm.scene.rotation.y = Math.PI; // Rotate model 180deg to face camera
        });
    },

    (progress) => console.log("Loading model...", 100.0 * (progress.loaded / progress.total), "%"),

    (error) => console.error(error)
);

// Animate Rotation Helper function
// 앉을때 양반다리 모양으로 접히는것이 rigRotation함수를 적용한 관절이 잘못된 것이 아닐까?
const rigRotation = (name, rotation = { x: 0, y: 0, z: 0 }, dampener = 1, lerpAmount = 0.5) => {
    if (!currentVrm) {
        return;
    }
    const Part = currentVrm.humanoid.getBoneNode(THREE.VRMSchema.HumanoidBoneName[name]);
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
const rigPosition = (name, position = { x: 0, y: 0, z: 0 }, dampener = 1, lerpAmount = 1) => {
    if (!currentVrm) {
        return;
    }
    const Part = currentVrm.humanoid.getBoneNode(THREE.VRMSchema.HumanoidBoneName[name]);
    if (!Part) {
        return;
    }
    let vector = new THREE.Vector3(position.x * dampener, position.y * dampener, position.z * dampener);
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
    riggedFace.eye.l = lerp(clamp(1 - riggedFace.eye.l, 0, 1), Blendshape.getValue(PresetName.Blink), 0.5);
    riggedFace.eye.r = lerp(clamp(1 - riggedFace.eye.r, 0, 1), Blendshape.getValue(PresetName.Blink), 0.5);
    riggedFace.eye = Kalidokit.Face.stabilizeBlink(riggedFace.eye, riggedFace.head.y);
    Blendshape.setValue(PresetName.Blink, riggedFace.eye.l);

    // Interpolate and set mouth blendshapes
    Blendshape.setValue(PresetName.I, lerp(riggedFace.mouth.shape.I, Blendshape.getValue(PresetName.I), 0.5));
    Blendshape.setValue(PresetName.A, lerp(riggedFace.mouth.shape.A, Blendshape.getValue(PresetName.A), 0.5));
    Blendshape.setValue(PresetName.E, lerp(riggedFace.mouth.shape.E, Blendshape.getValue(PresetName.E), 0.5));
    Blendshape.setValue(PresetName.O, lerp(riggedFace.mouth.shape.O, Blendshape.getValue(PresetName.O), 0.5));
    Blendshape.setValue(PresetName.U, lerp(riggedFace.mouth.shape.U, Blendshape.getValue(PresetName.U), 0.5));

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

        // rigPosition(
        //     "Hips", 
        //     {
        //         x: (pose3DLandmarks[23].x + pose3DLandmarks[24].x)/2,
        //         y: -1*(pose3DLandmarks[23].y + pose3DLandmarks[24].y)/2,
        //         z: (pose3DLandmarks[23].z + pose3DLandmarks[24].z)/2
        //     },
        //     1,
        //     0.9
        //     );

        // rigPosition(
        //     "Chest", 
        //     {
        //         x: (pose3DLandmarks[11].x + pose3DLandmarks[12].x),
        //         y: -1*(pose3DLandmarks[11].y + pose3DLandmarks[12].y),
        //         z: (pose3DLandmarks[11].z + pose3DLandmarks[12].z)
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "Spine", 
        //     {
        //         x: (pose3DLandmarks[11].x + pose3DLandmarks[12].x + pose3DLandmarks[23].x + pose3DLandmarks[24].x)/4,
        //         y: -1*((pose3DLandmarks[11].y + pose3DLandmarks[12].y)/2 + (pose3DLandmarks[23].y + pose3DLandmarks[24].y)/2)/3,
        //         z: (pose3DLandmarks[11].z + pose3DLandmarks[12].z + pose3DLandmarks[23].z + pose3DLandmarks[24].z)/4
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "RightUpperArm", 
        //     {
        //         x: pose3DLandmarks[11].x,
        //         y: -1*pose3DLandmarks[11].y,
        //         z: pose3DLandmarks[11].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "LeftUpperArm", 
        //     {
        //         x: pose3DLandmarks[12].x,
        //         y: -1*pose3DLandmarks[12].y,
        //         z: pose3DLandmarks[12].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "RightLowerArm", 
        //     {
        //         x: pose3DLandmarks[13].x,
        //         y: -1*pose3DLandmarks[13].y,
        //         z: pose3DLandmarks[13].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "LeftLowerArm", 
        //     {
        //         x: pose3DLandmarks[14].x,
        //         y: -1*pose3DLandmarks[14].y,
        //         z: pose3DLandmarks[14].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "RightHand", 
        //     {
        //         x: pose3DLandmarks[15].x,
        //         y: -1*pose3DLandmarks[15].y,
        //         z: pose3DLandmarks[15].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "LeftHand", 
        //     {
        //         x: pose3DLandmarks[16].x,
        //         y: -1*pose3DLandmarks[16].y,
        //         z: pose3DLandmarks[16].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "RightUpperLeg", 
        //     {
        //         x: pose3DLandmarks[23].x,
        //         y: -1*pose3DLandmarks[23].y,
        //         z: pose3DLandmarks[23].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "LeftUpperLeg", 
        //     {
        //         x: pose3DLandmarks[24].x,
        //         y: -1*pose3DLandmarks[24].y,
        //         z: pose3DLandmarks[24].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "RightLowerLeg", 
        //     {
        //         x: pose3DLandmarks[25].x,
        //         y: -1*pose3DLandmarks[25].y,
        //         z: pose3DLandmarks[25].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "LeftLowerLeg", 
        //     {
        //         x: pose3DLandmarks[26].x,
        //         y: -1*pose3DLandmarks[26].y,
        //         z: pose3DLandmarks[26].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "RightFoot", 
        //     {
        //         x: pose3DLandmarks[27].x,
        //         y: -1*pose3DLandmarks[27].y,
        //         z: pose3DLandmarks[27].z
        //     },
        //     1,
        //     0.9
        // );

        // rigPosition(
        //     "LeftFoot", 
        //     {
        //         x: pose3DLandmarks[28].x,
        //         y: -1*pose3DLandmarks[28].y,
        //         z: pose3DLandmarks[28].z
        //     },
        //     1,
        //     0.9
        // );

        // 엉덩이 관절 노드를 얻기
        const hipsNode = vrm.humanoid.getBoneNode(THREE.VRMSchema.HumanoidBoneName.Hips);
        // 노드의 위치 벡터를 가져오기
        const hipsPosition = new THREE.Vector3();
        if (hipsNode) {
            hipsPosition.setFromMatrixPosition(hipsNode.matrixWorld);
        }

        // 오른쪽 발 관절 노드를 얻기
        const rightFootNode = vrm.humanoid.getBoneNode(THREE.VRMSchema.HumanoidBoneName.RightFoot);
        // 노드의 위치 벡터를 저장할 변수
        const rightFootPosition = new THREE.Vector3();
        if (rightFootNode) {
            rightFootPosition.setFromMatrixPosition(rightFootNode.matrixWorld);

        }

        // 왼쪽 발 관절 노드를 얻기
        const leftFootNode = vrm.humanoid.getBoneNode(THREE.VRMSchema.HumanoidBoneName.LeftFoot);
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
                    x: 0,
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
                    x: 0,
                    y: hipsPosition.y - rightFootPosition.y + 0.13,
                    z: 0,
                },
                1,
                1
            );
        }

        rigRotation("Hips", riggedPose.Hips.rotation);

        console.log(vrm);
        console.log(riggedPose);
        riggedPose.LeftUpperLeg.y = riggedPose.LeftUpperLeg.y*-0.2;
        riggedPose.RightUpperLeg.y = riggedPose.RightUpperLeg.y*-0.2;
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
        rigRotation("LeftMiddleIntermediate", riggedLeftHand.LeftMiddleIntermediate);
        rigRotation("LeftMiddleDistal", riggedLeftHand.LeftMiddleDistal);
        rigRotation("LeftThumbProximal", riggedLeftHand.LeftThumbProximal);
        rigRotation("LeftThumbIntermediate", riggedLeftHand.LeftThumbIntermediate);
        rigRotation("LeftThumbDistal", riggedLeftHand.LeftThumbDistal);
        rigRotation("LeftLittleProximal", riggedLeftHand.LeftLittleProximal);
        rigRotation("LeftLittleIntermediate", riggedLeftHand.LeftLittleIntermediate);
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
        rigRotation("RightIndexIntermediate", riggedRightHand.RightIndexIntermediate);
        rigRotation("RightIndexDistal", riggedRightHand.RightIndexDistal);
        rigRotation("RightMiddleProximal", riggedRightHand.RightMiddleProximal);
        rigRotation("RightMiddleIntermediate", riggedRightHand.RightMiddleIntermediate);
        rigRotation("RightMiddleDistal", riggedRightHand.RightMiddleDistal);
        rigRotation("RightThumbProximal", riggedRightHand.RightThumbProximal);
        rigRotation("RightThumbIntermediate", riggedRightHand.RightThumbIntermediate);
        rigRotation("RightThumbDistal", riggedRightHand.RightThumbDistal);
        rigRotation("RightLittleProximal", riggedRightHand.RightLittleProximal);
        rigRotation("RightLittleIntermediate", riggedRightHand.RightLittleIntermediate);
        rigRotation("RightLittleDistal", riggedRightHand.RightLittleDistal);
    }
};

/* SETUP MEDIAPIPE HOLISTIC INSTANCE */
let videoElement = document.querySelector(".input_video"),
    guideCanvas = document.querySelector("canvas.guides");


//AJAX 좌표 전송



const onResults = (results) => {
    // Draw landmark guides
    drawResults(results);
    // Animate model
    animateVRM(currentVrm, results);



    if (results.poseLandmarks && results.poseLandmarks.length >= 27) {
    // 객체로 전송시 JSON.Stringify포함,  아래는 허리디스크 모델을 돌리기 위한 좌표 전송 코드
    
    const rightShoulderIndex =  11;
    const leftShoulderIndex =  12;
    const rightHipIndex =  23;
    const leftHipIndex = 24;
    const rightKneeIndex = 25;
    const leftKneeIndex = 26;
    const rightAnkleIndex = 27;
    const leftAnkleIndex = 28;

    var rightShoulderLandmark = results.poseLandmarks[rightShoulderIndex];
    var leftShoulderLandmark = results.poseLandmarks[leftShoulderIndex];
    var rightHipLandmark = results.poseLandmarks[rightHipIndex];
    var leftHipLandmark = results.poseLandmarks[leftHipIndex];
    var rightKneeLandmark = results.poseLandmarks[rightKneeIndex];
    var leftKneeLandmark = results.poseLandmarks[leftKneeIndex];
    var rightAnkleLandmark = results.poseLandmarks[rightAnkleIndex];
    var leftAnkleLandmark = results.poseLandmarks[leftAnkleIndex];


    //객체로 전송시
    var landmarkData = JSON.stringify(leftKneeLandmark);


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

    var CoordinateData = {
                    "rightShoulderXCoordinate": parseFloat(rightShoulderXCoordinate),
                    "rightShoulderYCoordinate": parseFloat(rightShoulderYCoordinate),
                    "rightShoulderZCoordinate": parseFloat(rightShoulderZCoordinate),

                    "leftShoulderXCoordinate": parseFloat(leftShoulderXCoordinate),
                    "leftShoulderYCoordinate": parseFloat(leftShoulderYCoordinate),
                    "leftShoulderZCoordinate": parseFloat(leftShoulderZCoordinate),

                    "rightHipXCoordinate": parseFloat(rightHipXCoordinate),
                    "rightHipYCoordinate": parseFloat(rightHipYCoordinate),
                    "rightHipZCoordinate": parseFloat(rightHipZCoordinate),

                    "leftHipXCoordinate": parseFloat(leftHipXCoordinate),
                    "leftHipYCoordinate": parseFloat(leftHipYCoordinate),
                    "leftHipZCoordinate": parseFloat(leftHipZCoordinate),

                    "rightKneeXCoordinate": parseFloat(rightKneeXCoordinate),
                    "rightKneeYCoordinate": parseFloat(rightKneeYCoordinate),
                    "rightKneeZCoordinate": parseFloat(rightKneeZCoordinate),

                    "leftKneeXCoordinate": parseFloat(leftKneeXCoordinate),
                    "leftKneeYCoordinate": parseFloat(leftKneeYCoordinate),
                    "leftKneeZCoordinate": parseFloat(leftKneeZCoordinate),

                    "rightAnkleXCoordinate": parseFloat(rightAnkleXCoordinate),
                    "rightAnkleYCoordinate": parseFloat(rightAnkleYCoordinate),
                    "rightAnkleZCoordinate": parseFloat(rightAnkleZCoordinate),

                    "leftAnkleXCoordinate": parseFloat(leftAnkleXCoordinate),
                    "leftAnkleYCoordinate": parseFloat(leftAnkleYCoordinate),
                    "leftAnkleZCoordinate": parseFloat(leftAnkleZCoordinate),
                    //"landmarkData": landmarkData,
    };
    console.log(CoordinateData);

    //보안 csrftoken
    var csrftoken = document.querySelector("meta[name=csrf_token]").content

    $.ajax({
        url: jsonUrl, // URL을 템플릿 태그로 설정
        type: 'POST',
        headers: {
            'X-CSRFToken': csrftoken, // CSRF 토큰 설정
        },
        data: CoordinateData,

        success: function(data) {
            console.log('Data sent successfully');
            // Class 저장 리스트
            var className = ['good_stand', 'good_progress', 'good_sit',
                             'knee_narrow_progress', 'knee_narrow_sit',
                             'knee_wide_progress', 'knee_wide_sit']
            
            // 리스트 형태로 예측확률 결과 저장
            var jsonDataPreList = [data.json_data0, data.json_data1, data.json_data2,
                                   data.json_data3, data.json_data4, data.json_data5,
                                   data.json_data6];

            var jsonAccuracy = data.json_data7;
            var jsonCnt = data.json_data9;
            var jsonClassIdx = data.json_data13;
            

            // jsonAccuracy가 90 이상이면 effect js 로드
            if (parseFloat(jsonAccuracy) >= 0.6) {
                // 외부 JavaScript 파일 로드
                var script = document.createElement('script');
                script.src = 'FullStack/sweaty/avatar/motion_effect/firework.js'; // 스크립트 파일 경로 지정
                document.head.appendChild(script);
            }s

            //HTML에 텍스트 로드
            var receivedDataElement0 = document.getElementById('showClass'); // HTML 요소 선택
            receivedDataElement0.innerHTML = 'Class: ' + className[parseInt(jsonClassIdx)]; // HTML
            var receivedDataElement1 = document.getElementById('showPre'); // HTML 요소 선택
            receivedDataElement1.innerHTML = 'Predict: ' + jsonDataPreList[parseInt(jsonClassIdx)]; // HTML
            var receivedDataElement2 = document.getElementById('showCnt'); // HTML 요소 선택
            receivedDataElement2.innerHTML = 'Count: ' + jsonCnt; // HTML
            var receivedDataElement3 = document.getElementById('showSquatAcuu'); // HTML 요소 선택
            receivedDataElement3.innerHTML = 'SquatAccu: ' + String(parseFloat(jsonAccuracy)*100) + '%'; // HTML
            

            //aframe 가상환경 안에  텍스트 로드
            //const shoulderText2 = document.querySelector('#shoulderText2');
            //shoulderText2.setAttribute('value', `probability1: (${JSON.stringify(jsonData)})`);




        },
        error: function(xhr, status, error) {
            console.error('Error sending data:', error);
                            // 오류가 발생한 경우 처리
            }
        });

};
}

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
    refineFaceLandmarks: true,
});
// Pass holistic a callback function
holistic.onResults(onResults);

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
        drawLandmarks(canvasCtx, [results.faceLandmarks[468], results.faceLandmarks[468 + 5]], {
            color: "#ffe603",
            lineWidth: 2,
        });
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

// Use `Mediapipe` utils to get camera - lower resolution = higher fps
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await holistic.send({ image: videoElement });
    },
    width: 640,
    height: 480,
    facingMode: 'environment'   //학교에서 빌린 웹캠은 후면카메라로 인식되어서 코드 추가함
});
camera.start();