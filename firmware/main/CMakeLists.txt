idf_component_register(
    SRCS "main.c" "infer.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_usb_serial_jtag esp_timer
)

set(MODEL_BIN "${CMAKE_CURRENT_LIST_DIR}/../../models/model.bin")
if(NOT EXISTS "${MODEL_BIN}")
    message(FATAL_ERROR
        "models/model.bin not found.\n"
        "Export a trained checkpoint first:\n"
        "  scripts/build_firmware.sh [checkpoint]\n"
        "or:\n"
        "  python tools/export.py checkpoints/best.pt --out models/model.bin "
        "--tokenizer models/tokenizer --dtype int8 --check")
endif()

target_add_binary_data(${COMPONENT_LIB} "${MODEL_BIN}" BINARY)
