//
// Created by elder on 8/14/2026.
//

#include "../Header/glUtils.cuh"

#include "../../Tiled/deviceFunctions.cuh"


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

const char *makeVertexShader() {
    return R"GLSL(
        #version 330 core

        layout (location = 0) vec2 aPos;
        layout (location = 1) vec2 aVel;

        out vec3 aColor;

        void main() {

        }
    )GLSL";
}

const char *makeFragmentShader() {
    return R"GLSL(
        #version 330 core

        in vec3 vertexColor;

        void main() {

        }
    )GLSL";
}

void setVAO(GLuint &VAO, GLuint &VBO, GLenum drawHint) {

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(VBO, N_BOIDS * sizeof(Boid), nullptr, drawHint);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Boid), reinterpret_cast<void *>(offsetof(Boid, position)));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Boid), reinterpret_cast<void *>(offsetof(Boid, velocity)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

GLuint compileProgram(const char *shader, const GLenum shaderType) {
    GLuint s = glCreateShader(shaderType);
    glShaderSource(s, 1, &shader, nullptr);
    glCompileShader(s);

    return s;
}