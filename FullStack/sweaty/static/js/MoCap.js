const remap = Kalidokit.Utils.remap;
const clamp = Kalidokit.Utils.clamp;
const lerp = Kalidokit.Vector.lerp;

/* THREEJS WORLD SETUP */
let currentVrm;

let startTime;
let endTime;

// renderer
const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// camera
const orbitCamera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
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
const orbitControls = new THREE.OrbitControls(orbitCamera, renderer.domElement);
orbitControls.screenSpacePanning = true;
orbitControls.target.set(0.0, 1.2, 0.0);
orbitControls.update();

// scene
const scene = new THREE.Scene();

// light
const light = new THREE.DirectionalLight(0xffffff);
light.position.set(2.0, 2.0, 1.0).normalize();
light.intensity = 1.5; // 밝기 조절
scene.add(light);

// const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
// scene.add(ambientLight);

// Main Render Loop
const clock = new THREE.Clock();

let mixer;
let mixer2;
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
    }

    if (mixer2) {
        mixer2.update(clock.getDelta());
    }

    renderer.render(scene, orbitCamera);
}
animate();

const loader = new THREE.GLTFLoader();

// Desert 맵 로드 함수
function loadDesert(MapUrl) {
    loader.load(MapUrl, (gltf) => {
        // 이전 맵 제거
        // if (currentMap) {
        //     scene.remove(currentMap);
        //     currentMap = null;
        // }

        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-5, -0.2, -40);
        mesh.scale.set(2, 2, 2);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("맵을 로드했습니다.");
    });
}

// Pyramid & Sphinx asset 로드 함수
function loadPyramid(MapUrl2) {
    loader.load(MapUrl2, (gltf) => {
        // 이전 맵 제거
        // if (currentMap) {
        //     scene.remove(currentMap);
        //     currentMap = null;
        // }

        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-50, 0, -20);
        mesh.scale.set(0.03, 0.03, 0.03);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("asset을 로드했습니다.");
    });
}

// Island 맵 로드 함수
function loadIsland(MapUrl) {
    loader.load(MapUrl, (gltf) => {
        // 이전 맵 제거
        // if (currentMap) {
        //     scene.remove(currentMap);
        //     currentMap = null;
        // }

        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-2, -1, 0);
        mesh.scale.set(1.5, 1.5, 1.5);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("맵을 로드했습니다.");
    });
}

// Island Maui asset 로드 함수
function loadMaui(MapUrl2) {
    loader.load(MapUrl2, (gltf) => {
        // 이전 맵 제거
        // if (currentMap) {
        //     scene.remove(currentMap);
        //     currentMap = null;
        // }

        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-2, 0, 0);
        mesh.scale.set(1.5, 1.5, 1.5);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("asset을 로드했습니다.");
    });
}

// Mountain 맵 로드 함수
function loadMountain(MapUrl) {
    loader.load(MapUrl, (gltf) => {
        // 이전 맵 제거
        // if (currentMap) {
        //     scene.remove(currentMap);
        //     currentMap = null;
        // }

        // 새 맵 로드
        const mesh = gltf.scene;
        mesh.position.set(-2.0, -47.0, 10);
        mesh.scale.set(2, 2, 2);
        scene.add(mesh);
        // currentMap = mesh;
        console.log("맵을 로드했습니다.");
    });
}

/* VRM CHARACTER SETUP */

// VRM 모델 로드 함수
function loadModel(modelUrl) {
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
            });
        },
        (progress) =>
            console.log(
                "모델 로딩 중...",
                100.0 * (progress.loaded / progress.total),
                "%"
            ),
        (error) => console.error(error)
    );
}

let trainer_action1;
let trainer_action2;
let trainer_action3;

loader.load("../static/assets/trainer.glb", (gltf) => {
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
});

// $("#loadModelButton1").click(function() {
//     loadModel(modelUrl);
// });

// $("#loadModelButton2").click(function() {
//     loadModel(modelUrl);
// });

// function loadMap1 (MapUrl){
//     loader.load(MapUrl, (gltf) => {
//     const mesh = gltf.scene;
//     mesh.position.set(-2.0, -47.0, 10);
//     mesh.scale.set(2, 2, 2);
//     scene.add(mesh);
//     currentMap = mesh;
// })

// }
// function loadMap2 (MapUrl){
//     loader.load(MapUrl, (gltf) => {
//     const mesh = gltf.scene;
//     mesh.position.set(0, -4, -10);
//     mesh.scale.set(10, 10 , 10);
//     scene.add(mesh);
//     currentMap = mesh;
// })

// }

// loadMap1 (MapUrl);

//  $("#image-button1").click(function() {
//     if (currentMap) {
//         scene.remove(currentMap);
//     }

//     loadMap1 (MapUrl);
//  });

//   $("#image-button2").click(function() {
//     if (currentMap) {
//         scene.remove(currentMap);

//     }

//     loadMap2 (MapUrl);
//  });

// ---------------------------------------
// ** 세트 수, 1세트당 횟수, 쉬는시간 변수 **

const TOTAL_SET = 2;
const TOTAL_NUM = 2;
const BREAK_TIME_NUM = 10; // 단위 : 초

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

// ------------------------------
// ** ajax 모듈 실행
var myxhr;
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

        success: function (data) {
            // Class 저장 리스트
            var className = [
                "good_stand",
                "good_progress",
                "good_sit",
                "knee_narrow_progress",
                "knee_narrow_sit",
                "knee_wide_progress",
                "knee_wide_sit",
            ];

            // 리스트 형태로 예측확률 결과 저장
            var jsonDataPreList = [
                data.json_data0,
                data.json_data1,
                data.json_data2,
                data.json_data3,
                data.json_data4,
                data.json_data5,
                data.json_data6,
            ];

            var jsonAccuracy = data.json_data7;
            var jsonCnt = data.json_data9;
            var jsonClassIdx = data.json_data13;
            var jsonSquatState = data.json_11;

            var processAccuracy = parseFloat(jsonAccuracy) * 100;

            //HTML에 텍스트 로드
            var receivedDataElement0 = document.getElementById("showClass"); // HTML 요소 선택
            receivedDataElement0.innerHTML =
                "Class: " + className[parseInt(jsonClassIdx)]; // HTML
            var receivedDataElement1 = document.getElementById("showPre"); // HTML 요소 선택
            receivedDataElement1.innerHTML =
                "Predict: " + jsonDataPreList[parseInt(jsonClassIdx)]; // HTML
            var receivedDataElement2 = document.getElementById("showCnt"); // HTML 요소 선택
            receivedDataElement2.innerHTML = "Count: " + jsonCnt; // HTML
            var receivedDataElement3 = document.getElementById("showSquatAcuu"); // HTML 요소 선택
            receivedDataElement3.innerHTML =
                "SquatAccu: " + String(processAccuracy) + "%"; // HTML

            // 프로세스 바에 표시할 로직
            if (processAccuracy > 0) {
                var receivedDataElement4 =
                    document.getElementById("text-process");
                receivedDataElement4.innerHTML = parseInt(processAccuracy);
            }

            // 현재 jsonCnt와 이전 jsonCnt를 비교하여 값이 변경되었는지 확인
            if (jsonCnt !== previousJsonCnt) {
                mixer2.stopAllAction();
                trainer_action3.setDuration(0.005);
                trainer_action3.play();
                loader.load("../../static/assets/flower_pink.gltf", (gltf) => {
                    const mesh = gltf.scene;
                    mesh.position.set(0.0, 1.5, -1.0);
                    mesh.scale.set(0.6, 0.6, 0.6);
                    scene.add(mesh);
                    mixer = new THREE.AnimationMixer(mesh);
                    const clips = gltf.animations;
                    const clip = THREE.AnimationClip.findByName(
                        clips,
                        "Rotate"
                    );
                    const action = mixer.clipAction(clip);

                    action.setDuration(0.005);
                    action.play();

                    // 1초 후에 GLTF를 삭제하는 함수 호출
                    setTimeout(() => {
                        scene.remove(mesh); // Scene에서 mesh 제거
                        mixer = null;
                        mixer2.stopAllAction();
                        trainer_action2.play();
                    }, 3000); // 1000ms = 1초
                });
                console.log("꽃생성");
                // 변경된 경우에만 처리 --------------------------
                previousJsonCnt = jsonCnt; // 이전 jsonCnt 업데이트

                // 이전 텍스트 메시지 제거
                const previousTextMesh = scene.getObjectByName("countText");
                scene.remove(previousTextMesh);

                // 새로운 텍스트 메시지 생성
                const fontLoader = new THREE.FontLoader();
                fontLoader.load(
                    "../../static/fonts/DungGeunMo_Regular.json",
                    function (font) {
                        const textGeometry = new THREE.TextGeometry(
                            "Count: " + jsonCnt,
                            {
                                font: font,
                                size: 1,
                                height: 0.1,
                                bevelEnabled: true, // 윤곽선 활성화
                                bevelSize: 0.05, // 윤곽선 크기
                                bevelThickness: 0.05, // 윤곽선 두께
                            }
                        );

                        const textMaterial = new THREE.MeshStandardMaterial({
                            color: 0x507a03,
                        }); // 텍스트 색상
                        const textMesh = new THREE.Mesh(
                            textGeometry,
                            textMaterial
                        );

                        // 윤곽선 색상을 검정색(0x000000)으로 설정
                        textMaterial.emissive.setHex(0x000000);
                        textMesh.name = "countText";
                        textMesh.position.set(-3, 3, 0);
                        scene.add(textMesh);
                    }
                );

                // 모델의 blendShapeProxy를 가져옴
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

                moveCameraToTarget(
                    { x: 0.0, y: 1.8, z: 1.4 }, // 목표 위치
                    { x: -1.0, y: 1.4, z: 0.0 }, // 목표 방향
                    1 // 1초 동안 이동
                );
                setTimeout(() => {
                    moveCameraDefault();
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

            if (
                parseFloat(jsonAccuracy) * 100 < 70 &&
                switchAccu == 0 &&
                jsonAccuracy !== previousJsonAccu
            ) {
                switchAccu = 1;
                mixer2.stopAllAction();
                trainer_action1.setDuration(0.002);
                trainer_action1.play();
                loader.load(
                    "../../static/assets/cloud_thunder.gltf",
                    (gltf) => {
                        // 변경된 경우에만 처리
                        previousJsonAccu = jsonAccuracy; // 이전 jsonCnt 업데이트

                        const mesh = gltf.scene;
                        mesh.position.set(0.0, 1.5, -1.0);
                        mesh.scale.set(0.5, 0.5, 0.5);
                        scene.add(mesh);
                        console.log("구름생성");
                        mixer = new THREE.AnimationMixer(mesh);
                        const clips = gltf.animations;
                        const clip1 = THREE.AnimationClip.findByName(
                            clips,
                            "Movement1"
                        );
                        const clip2 = THREE.AnimationClip.findByName(
                            clips,
                            "Movement2"
                        );
                        const clip3 = THREE.AnimationClip.findByName(
                            clips,
                            "Movement3"
                        );

                        const action1 = mixer.clipAction(clip1);
                        const action2 = mixer.clipAction(clip2);
                        const action3 = mixer.clipAction(clip3);

                        action1.setDuration(0.005);
                        action2.setDuration(0.005);
                        action3.setDuration(0.005);

                        action1.play();
                        action2.play();
                        action3.play();

                        // 1초 후에 GLTF를 삭제하는 함수 호출
                        setTimeout(() => {
                            scene.remove(mesh); // Scene에서 mesh 제거
                            mixer = null;
                            mixer2.stopAllAction();
                            trainer_action2.play();
                        }, 3000); // 1000ms = 1초
                        switchAccu = 0;
                    }
                );

                // 모델의 blendShapeProxy를 가져옴
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

                moveCameraToTarget(
                    { x: 0.0, y: 1.8, z: 1.4 }, // 목표 위치
                    { x: -1.0, y: 1.4, z: 0.0 }, // 목표 방향
                    1 // 1초 동안 이동
                );
                setTimeout(() => {
                    moveCameraDefault();
                }, 1000);
            }

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

const rgbeloader = new THREE.RGBELoader();

rgbeloader.load(
    "../../static/assets/kloofendal_48d_partly_cloudy_puresky_4k.hdr",
    (texture) => {
        texture.mapping = THREE.EquirectangularReflectionMapping;
        scene.background = texture;

        setTimeout(() => {
            // // 파일 로드 완료 후 로딩 화면 숨기기
            // loadingScreen.style.display = 'none';
            document.getElementById("loading-screen").style.display = "none";
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
                        // test
                        setBreakTime();
                    }, 4000);
                })
                .start(); // 애니메이션 시작
        }, 3000);
    }
);

const fontLoader = new THREE.FontLoader();
fontLoader.load("../../static/fonts/DungGeunMo_Regular.json", function (font) {
    const textGeometry = new THREE.TextGeometry("Count: 0", {
        font: font,
        size: 1,
        height: 0.1,
        bevelEnabled: true, // 윤곽선 활성화
        bevelSize: 0.05, // 윤곽선 크기
        bevelThickness: 0.05, // 윤곽선 두께
    });

    const textMaterial = new THREE.MeshStandardMaterial({ color: 0x507a03 }); // 텍스트 색상
    const textMesh = new THREE.Mesh(textGeometry, textMaterial);

    // 윤곽선 색상을 검정색(0x000000)으로 설정
    textMaterial.emissive.setHex(0x000000);
    textMesh.name = "countText";
    textMesh.position.set(-3, 3, 0);
    scene.add(textMesh);
});

let currentTween;

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
            orbitControls.target.set(
                object.lookAtX,
                object.lookAtY,
                object.lookAtZ
            );
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
            orbitControls.target.set(
                object.lookAtX,
                object.lookAtY,
                object.lookAtZ
            );
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
            orbitControls.target.set(
                object.lookAtX,
                object.lookAtY,
                object.lookAtZ
            );
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
            3 * 1000
        )
        .easing(TWEEN.Easing.Quadratic.InOut)
        .onUpdate((object) => {
            // 카메라의 위치와 방향을 업데이트
            orbitCamera.position.set(object.x, object.y, object.z);
            orbitControls.target.set(
                object.lookAtX,
                object.lookAtY,
                object.lookAtZ
            );
            orbitControls.update();
        });

    tween3.start();

    setTimeout(() => {
        currentTween.start();
    }, 3000);
}

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
            orbitControls.target.set(
                object.lookAtX,
                object.lookAtY,
                object.lookAtZ
            );
            orbitControls.update();
        })
        .start();
}

// Animate Rotation Helper function
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
        rigRotation(
            "LeftRingIntermediate",
            riggedLeftHand.LeftRingIntermediate
        );
        rigRotation("LeftRingDistal", riggedLeftHand.LeftRingDistal);
        rigRotation("LeftIndexProximal", riggedLeftHand.LeftIndexProximal);
        rigRotation(
            "LeftIndexIntermediate",
            riggedLeftHand.LeftIndexIntermediate
        );
        rigRotation("LeftIndexDistal", riggedLeftHand.LeftIndexDistal);
        rigRotation("LeftMiddleProximal", riggedLeftHand.LeftMiddleProximal);
        rigRotation(
            "LeftMiddleIntermediate",
            riggedLeftHand.LeftMiddleIntermediate
        );
        rigRotation("LeftMiddleDistal", riggedLeftHand.LeftMiddleDistal);
        rigRotation("LeftThumbProximal", riggedLeftHand.LeftThumbProximal);
        rigRotation(
            "LeftThumbIntermediate",
            riggedLeftHand.LeftThumbIntermediate
        );
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
        rigRotation(
            "RightRingIntermediate",
            riggedRightHand.RightRingIntermediate
        );
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

/* SETUP MEDIAPIPE HOLISTIC INSTANCE */
let videoElement = document.querySelector(".input_video"),
    guideCanvas = document.querySelector("canvas.guides");

//AJAX 좌표 전송

// 이전 jsonCnt를 저장할 변수 선언
let previousJsonCnt = "0";
let previousJsonAccu = "-1";
let switchAccu = 0;
console.log("초기화");

const onResults = (results) => {
    // Draw landmark guides
    drawResults(results);
    // Animate model
    animateVRM(currentVrm, results);

    if (results.poseLandmarks && results.poseLandmarks.length >= 27) {
        // 객체로 전송시 JSON.Stringify포함,  아래는 허리디스크 모델을 돌리기 위한 좌표 전송 코드

        const rightShoulderIndex = 11;
        const leftShoulderIndex = 12;
        const rightHipIndex = 23;
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
            rightShoulderXCoordinate: parseFloat(rightShoulderXCoordinate),
            rightShoulderYCoordinate: parseFloat(rightShoulderYCoordinate),
            rightShoulderZCoordinate: parseFloat(rightShoulderZCoordinate),

            leftShoulderXCoordinate: parseFloat(leftShoulderXCoordinate),
            leftShoulderYCoordinate: parseFloat(leftShoulderYCoordinate),
            leftShoulderZCoordinate: parseFloat(leftShoulderZCoordinate),

            rightHipXCoordinate: parseFloat(rightHipXCoordinate),
            rightHipYCoordinate: parseFloat(rightHipYCoordinate),
            rightHipZCoordinate: parseFloat(rightHipZCoordinate),

            leftHipXCoordinate: parseFloat(leftHipXCoordinate),
            leftHipYCoordinate: parseFloat(leftHipYCoordinate),
            leftHipZCoordinate: parseFloat(leftHipZCoordinate),

            rightKneeXCoordinate: parseFloat(rightKneeXCoordinate),
            rightKneeYCoordinate: parseFloat(rightKneeYCoordinate),
            rightKneeZCoordinate: parseFloat(rightKneeZCoordinate),

            leftKneeXCoordinate: parseFloat(leftKneeXCoordinate),
            leftKneeYCoordinate: parseFloat(leftKneeYCoordinate),
            leftKneeZCoordinate: parseFloat(leftKneeZCoordinate),

            rightAnkleXCoordinate: parseFloat(rightAnkleXCoordinate),
            rightAnkleYCoordinate: parseFloat(rightAnkleYCoordinate),
            rightAnkleZCoordinate: parseFloat(rightAnkleZCoordinate),

            leftAnkleXCoordinate: parseFloat(leftAnkleXCoordinate),
            leftAnkleYCoordinate: parseFloat(leftAnkleYCoordinate),
            leftAnkleZCoordinate: parseFloat(leftAnkleZCoordinate),
            //"landmarkData": landmarkData,
        };

        myxhr = getAjax(CoordinateData);
    }
};

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

let frameCounter = 0;

// Use `Mediapipe` utils to get camera - lower resolution = higher fps
const camera = new Camera(videoElement, {
    onFrame: async () => {
        // 5프레임마다 한 번씩만 비디오를 전송
        if (frameCounter % 1 === 0) {
            await holistic.send({ image: videoElement });
        }

        frameCounter++;
    },
    width: 640,
    height: 480,
    facingMode: "environment", //학교에서 빌린 웹캠은 후면카메라로 인식되어서 코드 추가함
});
camera.start();
