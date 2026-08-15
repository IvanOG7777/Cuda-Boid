//
// Created by elder on 8/14/2026.
//

#include "../Header/glUtils.cuh"


GLFWwindow *createWindow(int w, int h, const char* title) {
    if (w == 0||h == 0) {
        std:: cerr << "W OR H IS 0\n";
        exit(EXIT_FAILURE);
    }
    GLFWwindow *window = glfwCreateWindow(w, h, title, nullptr, nullptr);

    if (window == nullptr) {
        std:: cerr << "WINDOW IS NULL\n";
        exit(EXIT_FAILURE);
    }

    return window;
}