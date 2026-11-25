#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "infer.h"

static const char *TAG = "main";

// model.bin is embedded by target_add_binary_data() in CMakeLists.txt
extern const uint8_t model_bin_start[] asm("_binary_model_bin_start");
extern const uint8_t model_bin_end[] asm("_binary_model_bin_end");

static eslm_t *g_model;

static void usb_write(const void *data, size_t n) {
    usb_serial_jtag_write_bytes(data, n, pdMS_TO_TICKS(200));
}

static void usb_puts(const char *s) { usb_write(s, strlen(s)); }

static void usb_printf(const char *fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) usb_write(buf, (size_t)(n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1));
}

static void handle_gen(char *args) {
    int n = 64, top_k = 40, off = 0;
    float temp = 0.8f;
    int got = sscanf(args, " %d %f %n", &n, &temp, &off);
    if (got < 1) {
        n = 64;
        off = 0;
    }
    if (got < 2) {
        temp = 0.8f;
        off = 0;
        // second field may still be a prompt; try `GEN 50 <prompt>`
        char *sp = strchr(args, ' ');
        if (sp) off = (int)(sp - args) + 1;
    }
    if (n <= 0) n = 1;
    if (n > 512) n = 512;

    const char *prompt = args + off;
    while (*prompt == ' ') prompt++;
    if (!*prompt) prompt = "《";

    int ctx = eslm_ctx(g_model);
    int plen = (int)strlen(prompt);
    if (plen > ctx - 1) {
        prompt += plen - (ctx - 1);
        plen = ctx - 1;
    }

    eslm_reset(g_model);
    eslm_set_seed(g_model, (uint32_t)(esp_timer_get_time() & 0xffffffffu));

    int pos = 0;
    for (int i = 0; i < plen; i++) {
        eslm_forward(g_model, (unsigned char)prompt[i], pos++);
    }
    usb_write(prompt, (size_t)plen);

    int64_t t0 = esp_timer_get_time();
    int produced = 0;
    for (int i = 0; i < n && pos < ctx; i++) {
        int tok = eslm_sample(g_model, temp, top_k);
        unsigned char ch = (unsigned char)tok;
        usb_write(&ch, 1);
        eslm_forward(g_model, tok, pos++);
        produced++;
        if ((i & 7) == 7) vTaskDelay(1);  // keep the IDLE task (and WDT) alive
    }
    int64_t dt = esp_timer_get_time() - t0;

    float tps = dt > 0 ? produced * 1e6f / (float)dt : 0.0f;
    usb_printf("\r\n[OK] n=%d ms=%lld tok_s=%.1f\r\n", produced, (long long)(dt / 1000), tps);
    ESP_LOGI(TAG, "gen %d tok in %lld ms (%.1f tok/s)", produced, (long long)(dt / 1000), tps);
}


static void handle_chat(char *args) {
    int n = 96, off = 0;
    float temp = 0.8f;
    int got = sscanf(args, " %d %f %n", &n, &temp, &off);
    if (got < 1) { usb_puts("ERR chat args\r\n"); return; }
    if (got < 2) {
        temp = 0.8f;
        off = 0;
        char *sp = strchr(args, ' ');
        if (sp) off = (int)(sp - args) + 1;
    }
    if (n <= 0) n = 1;
    if (n > 400) n = 400;

    int ctx = eslm_ctx(g_model);
    int vocab = eslm_vocab(g_model);
    eslm_reset(g_model);
    eslm_set_seed(g_model, (uint32_t)(esp_timer_get_time() & 0xffffffffu));

    int pos = 0;
    char *p = args + off;
    while (*p && pos < ctx - 1) {
        while (*p == ' ') p++;
        if (!*p) break;
        char *end;
        long id = strtol(p, &end, 10);
        if (end == p) break;
        if (id >= 0 && id < vocab) eslm_forward(g_model, (int)id, pos++);
        p = end;
    }

    int64_t t0 = esp_timer_get_time();
    int produced = 0, e = eslm_special_e(g_model);
    for (int i = 0; i < n && pos < ctx; i++) {
        int tok = eslm_sample(g_model, temp, 40);
        if (tok == e) break;
        if (!eslm_is_special(g_model, tok)) {
            const uint8_t *bytes;
            size_t blen = eslm_token_bytes(g_model, tok, &bytes);
            if (blen) usb_write(bytes, blen);
        }
        eslm_forward(g_model, tok, pos++);
        produced++;
        if ((i & 7) == 7) vTaskDelay(1);
    }
    int64_t dt = esp_timer_get_time() - t0;
    float tps = dt > 0 ? produced * 1e6f / (float)dt : 0.0f;
    usb_printf("\r\n[OK] n=%d ms=%lld tok_s=%.1f\r\n", produced, (long long)(dt / 1000), tps);
    ESP_LOGI(TAG, "chat %d tok in %lld ms (%.1f tok/s)", produced, (long long)(dt / 1000), tps);
}

static void handle_line(char *line) {
    if (!strncmp(line, "PING", 4)) {
        usb_printf("PONG eslm fw=1 ctx=%d\r\n", eslm_ctx(g_model));
    } else if (!strncmp(line, "INFO", 4)) {
        usb_printf("INFO params=%d ctx=%d dtype=%s free_int=%uKB free_psram=%uKB\r\n",
                   eslm_params(g_model), eslm_ctx(g_model),
                   eslm_dtype(g_model) ? "int8" : "f32",
                   (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                   (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    } else if (!strncmp(line, "GEN", 3)) {
        handle_gen(line + 3);
    } else if (!strncmp(line, "CHAT", 4)) {
        handle_chat(line + 4);
    } else if (!strncmp(line, "RESET", 5)) {
        eslm_reset(g_model);
        usb_puts("OK\r\n");
    } else if (!strncmp(line, "STATS", 5)) {
        usb_printf("STATS free_int=%uKB free_psram=%uKB\r\n",
                   (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                   (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    } else if (*line) {
        usb_printf("ERR unknown\r\n");
    }
}

void app_main(void) {
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.tx_buffer_size = 4096;
    cfg.rx_buffer_size = 4096;
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));

    size_t mlen = (size_t)(model_bin_end - model_bin_start);
    ESP_LOGI(TAG, "model blob %u bytes", (unsigned)mlen);
    g_model = eslm_init(model_bin_start, mlen);
    if (!g_model) {
        ESP_LOGE(TAG, "model init failed");
        usb_puts("ERR model init failed\r\n");
        return;
    }

    usb_printf("\r\neslm ready: %d params, ctx %d, weights in %s\r\n",
               eslm_params(g_model), eslm_ctx(g_model),
               heap_caps_get_free_size(MALLOC_CAP_SPIRAM) > 4096 ? "PSRAM" : "flash");

    static char line[1024];
    size_t len = 0;
    for (;;) {
        uint8_t tmp[128];
        int n = usb_serial_jtag_read_bytes(tmp, sizeof(tmp), pdMS_TO_TICKS(100));
        for (int i = 0; i < n; i++) {
            char ch = (char)tmp[i];
            if (ch == '\r') continue;
            if (ch == '\n') {
                line[len] = 0;
                handle_line(line);
                len = 0;
            } else if (len < sizeof(line) - 1) {
                line[len++] = ch;
            }
        }
    }
}
