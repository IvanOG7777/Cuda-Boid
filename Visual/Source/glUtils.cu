//
// Created by elder on 8/14/2026.
//

#include <iostream>

#include "../Header/glUtils.cuh"
#include "../Header/deviceFunctions.cuh"


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

        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aVel;

        out vec3 vertexColor;

        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            gl_PointSize = 2.0;
            vertexColor = vec3(1.0, 1.0, 1.0);
        }
    )GLSL";
}

const char *makeFragmentShader() {
    return R"GLSL(
        #version 330 core

        in vec3 vertexColor;

        out vec4 FragColor;

        void main() {
            FragColor = vec4(vertexColor, 1.0);
        }
    )GLSL";
}

void setVAO(GLuint &VAO, GLuint &VBO, GLenum drawHint) {

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, N_BOIDS * sizeof(Boid), nullptr, drawHint);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Boid), reinterpret_cast<void *>(offsetof(Boid, position)));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Boid), reinterpret_cast<void *>(offsetof(Boid, velocity)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

GLuint compileShader(const char *shader, const GLenum shaderType) {
    GLuint s = glCreateShader(shaderType);
    glShaderSource(s, 1, &shader, nullptr);
    glCompileShader(s);

    return s;
}