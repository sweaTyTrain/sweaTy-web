document.addEventListener("DOMContentLoaded", () => {
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

  /** Begin Firework **/
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
  /** End Firework **/
  /** Begin Spark **/
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

          if (elapsedTime < 10) { // 10초 동안 애니메이션 실행
              // 오디오 요소 생성 및 재생 (한 번만 재생)
              if (elapsedTime < 0.1) {
                  var audio = new Audio('FullStack/sweaty/avatar/motion_effect/mp3-output-ttsfree(dot)com (1).mp3');
                  audio.play();
              }

              if (!document.getElementById("exerciseAccuracyText")) {
                  // "운동 정확도가 90% 넘었어요!" 텍스트 추가
                  let textElement = document.createElement("div");
                  textElement.id = "exerciseAccuracyText";
                  textElement.textContent = "운동 정확도가 90% 넘었어요!";
                  textElement.style.position = "absolute";
                  textElement.style.top = "50%";
                  textElement.style.left = "50%";
                  textElement.style.transform = "translate(-50%, -50%)";
                  textElement.style.color = "white";
                  textElement.style.fontSize = "35px";
                  textElement.style.width = "300px";
                  textElement.style.textAlign = "center";
                  textElement.style.whiteSpace = "nowrap";
                  textElement.style.fontWeight = "bold";

                  document.body.appendChild(textElement);
              }

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
          } else {
              // 애니메이션이 종료되면
              fireworkArray = [];
              sparkArray = [];
              document.body.removeChild(canvas);

              let textElement = document.getElementById("exerciseAccuracyText");
              if (textElement) {
                  document.body.removeChild(textElement);
              }
          }
      }

      frame(); // frame 함수 호출
  }

  animate();
});
