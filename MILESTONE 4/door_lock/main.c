/*
 * DOOR LOCK - ESP32 Arduino Core 3.x / ESP-IDF v5 Compatible
 * Nhan dien khuon mat mo cua
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_log.h"

#define SERVO_GPIO 13
#define LEDC_TIMER LEDC_TIMER_0
#define LEDC_MODE LEDC_LOW_SPEED_MODE    // Đã đổi sang LOW_SPEED_MODE để tương thích ESP-IDF v5
#define LEDC_OUTPUT_LED LEDC_CHANNEL_0
#define LEDC_DUTY_RES LEDC_TIMER_10_BIT
#define LEDC_FREQUENCY 50

static const char *TAG = "DOOR_LOCK";

// Goc servo
#define DOOR_LOCK_ANGLE 0
#define DOOR_OPEN_ANGLE 90

void setup_servo(void) {
    ledc_timer_config_t ledc_timer = {
        .speed_mode = LEDC_MODE,
        .duty_resolution = LEDC_DUTY_RES,
        .timer_num = LEDC_TIMER,
        .freq_hz = LEDC_FREQUENCY,
        .clk_cfg = LEDC_AUTO_CLK
    };
    ledc_timer_config(&ledc_timer);

    ledc_channel_config_t ledc_channel = {
        .gpio_num = SERVO_GPIO,
        .speed_mode = LEDC_MODE,
        .channel = LEDC_OUTPUT_LED,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = LEDC_TIMER,
        .duty = 0,
        .hpoint = 0
        // Đã XÓA dòng .intr_alloc_flags gây lỗi
    };
    ledc_channel_config(&ledc_channel);
}

void set_servo_angle(uint32_t angle) {
    // PWM 50Hz -> period 20000us. 10-bit resolution -> 0..1023
    // 0 deg -> 500us (~25 duty), 180 deg -> 2500us (~128 duty)
    if (angle > 180) angle = 180;
    uint32_t pulse_us = 500 + (angle * 2000) / 180;
    uint32_t duty = (pulse_us * 1023) / 20000;
    ledc_set_duty(LEDC_MODE, LEDC_OUTPUT_LED, duty);
    ledc_update_duty(LEDC_MODE, LEDC_OUTPUT_LED);
}

void open_door(void) {
    ESP_LOGI(TAG, "Door opened! Will auto-lock in 5 seconds.");
    set_servo_angle(DOOR_OPEN_ANGLE);
    vTaskDelay(pdMS_TO_TICKS(5000));
    set_servo_angle(DOOR_LOCK_ANGLE);
    ESP_LOGI(TAG, "Door locked!");
}

// Chuyển app_main() thành setup() và loop() chuẩn Arduino
void setup() {
    Serial.begin(115200);
    
    setup_servo();
    set_servo_angle(DOOR_LOCK_ANGLE);
    
    ESP_LOGI(TAG, "ESP32 Door Lock System Ready");
    ESP_LOGI(TAG, "Waiting for OPEN command...");
}

void loop() {
    char buffer[32];
    if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
        if (strstr(buffer, "OPEN") != NULL) {
            ESP_LOGI(TAG, "Received OPEN command - Unlocking door!");
            open_door();
        }
    }
    vTaskDelay(pdMS_TO_TICKS(100));
}