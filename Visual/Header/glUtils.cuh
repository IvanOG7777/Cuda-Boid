//
// Created by elder on 8/14/2026.
//

#ifndef CUDAPRACTICE_GLUTILS_CUH
#define CUDAPRACTICE_GLUTILS_CUH

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>


GLFWwindow *createWindow(int w, int h, const char* title);

const char* makeVertexShader();
const char* makeFragmentShader();

void setVAO(GLuint &VAO, GLuint &VBO, GLenum drawHint);

GLuint compileShader(const char *shader, GLenum shaderType);

#endif //CUDAPRACTICE_GLUTILS_CUH