#include <Mouse.h>

void setup() {
  Serial.begin(9600);
  Mouse.begin();
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      int deltaX = data.substring(0, commaIndex).toInt();
      int deltaY = data.substring(commaIndex + 1).toInt();
      Mouse.move(deltaX, deltaY);
    }
  }
}
