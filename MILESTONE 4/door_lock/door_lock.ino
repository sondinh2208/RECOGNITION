/*
 * DOOR LOCK - Nhan dien khuon mat mo cua
 * Ket noi ESP32 voi servo motor de mo cua khi nhan dien dung
 */

#if defined(ESP32)
  #include <ESP32Servo.h>
#else
  #include <Servo.h>
#endif

// Cau hinh Servo
Servo doorServo;
int servoPin = 13;  // Chan GPIO dieu khien servo
int doorLockPos = 0;   // Vi tri khoa (0-180)
int doorOpenPos = 90;  // Vi tri mo cua (0-180)

// Bien dem thoi gian
unsigned long openTime = 0;
bool doorOpened = false;

void setup() {
  Serial.begin(9600);
  doorServo.attach(servoPin);
  doorServo.write(doorLockPos);  // Khoa ban dau
  
  Serial.println("ESP32 Door Lock System Ready");
  Serial.println("Waiting for OPEN command from PC...");
}

void loop() {
  // Kiem tra neu co du lieu tu Serial
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "OPEN") {
      Serial.println("Received OPEN command - Unlocking door!");
      openDoor();
    }
  }
  
  // Tu dong khoa lai sau 5 giay
  if (doorOpened && (millis() - openTime > 5000)) {
    lockDoor();
  }
}

void openDoor() {
  doorServo.write(doorOpenPos);
  doorOpened = true;
  openTime = millis();
  Serial.println("Door opened! Will auto-lock in 5 seconds.");
}

void lockDoor() {
  doorServo.write(doorLockPos);
  doorOpened = false;
  Serial.println("Door locked!");
}