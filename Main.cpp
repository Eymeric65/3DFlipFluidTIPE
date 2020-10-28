#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include <iostream>
#include <vector>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include <math.h>

#include "Visual/Shader.h"
#include "Visual/Camera.h"

#include "Visual/GeoFunc.h"

#include "CLASSsim/FLIPimpl.h"

//#include<sphere.cpp>

//#define ONESTEPSIM


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// settings
unsigned int SCR_WIDTH = 1001;
unsigned int SCR_HEIGHT = 1000;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

int main()
{


    // a ne pas modifier ----------------------------------------------------------------------------------------------------
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_REFRESH_RATE, GL_DONT_CARE);

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSwapInterval(0); // pas de limite d'image

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    // fin  ----------------------------------------------------------------------------------------------------

    Shader firstshader("shader/vertexbase.vs", "shader/fragmentbase.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------

    std::vector<GLfloat> v_vertex_data;
    std::vector<GLuint> v_indices;

    int vertcount = spherecreate(v_vertex_data, v_indices, 5, 5);

    std::vector<glm::vec3> position;

    ///std::cout << vertcount << std::endl;

    int index = 0;
    float offset = 0.0f;
    for (int x = 0; x < 20; x += 1)
    {
        for (int y = 0; y < 20; y += 1)
        {
            for (int z = 0; z < 20; z += 1)
            {
                glm::vec3 translation;
                translation.x = (float)x / 2.0f + 1.5f;
                translation.y = (float)y / 2.0f +1.5f;
                translation.z = (float)z / 2.0f + 1.5f;

                position.push_back(translation);

            }
        }
    }

    const int PartCount = position.size();


    FlipSim FlipEngine(40.0, 20.0, 20.0, 1.0, PartCount,0.001);


    GLuint particles_position_buffer;
    glGenBuffers(1, &particles_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
    // Initialize with empty (NULL) buffer : it will be updated later, each frame.
    glBufferData(GL_ARRAY_BUFFER, position.size() * 3 * sizeof(float), position.data(), GL_DYNAMIC_DRAW);


    glBindBuffer(GL_ARRAY_BUFFER, 0);

    FlipEngine.linkPos(particles_position_buffer);

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, v_vertex_data.size() * sizeof(GLfloat), v_vertex_data.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, v_indices.size() * sizeof(GLuint), v_indices.data(), GL_STATIC_DRAW);



    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
    glVertexAttribDivisor(1, 1); // positions : one per quad (its center) -> 1

    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind


    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    //glBindVertexArray(0);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_FRONT);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW);


    int FPSlimiter = 0;
    
    FlipEngine.StartCompute();

    FlipEngine.TransferToGrid();

    FlipEngine.AddExternalForces();

    FlipEngine.Boundaries();

    FlipEngine.PressureCompute();

    FlipEngine.AddPressure();

    FlipEngine.TransferToParticule();

    FlipEngine.Integrate();

    FlipEngine.EndCompute();
    

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {

        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        FPSlimiter++;


        if (FPSlimiter > 100)
        {
            FPSlimiter = 0;
            char fps[64];
            int fpsfs = snprintf(fps, sizeof fps, "%f", 1 / deltaTime);
            glfwSetWindowTitle(window, fps);
        }


        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // draw our first triangle
        //glUseProgram(shaderProgram);

        firstshader.use();

        glm::mat4 model = glm::mat4(1.0f);
        //glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        glm::mat4 view = camera.GetViewMatrix();


        view = glm::translate(view, glm::vec3(-20.0f, -20.0f, -30.0f));

        projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 10000.0f);

        int projLoc = glGetUniformLocation(firstshader.ID, "projection");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        int modelLoc = glGetUniformLocation(firstshader.ID, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));


        int viewLoc = glGetUniformLocation(firstshader.ID, "view");

        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

        int lightcolor = glGetUniformLocation(firstshader.ID, "lightColor");

        int Objectcolor = glGetUniformLocation(firstshader.ID, "objectColor");
        glUniform3f(lightcolor, 1.0f, 1.0f, 1.0f);


        glUniform3f(Objectcolor, 0.7f, 0.0f, 0.0f);

        int lightPos = glGetUniformLocation(firstshader.ID, "lightPos");
        glUniform3f(lightPos, 0.0f, 2.0f, 2.0f);

#ifdef ONESTEPSIM
#else
        
        FlipEngine.StartCompute();

        FlipEngine.TransferToGrid();


        FlipEngine.AddExternalForces();
        
        FlipEngine.Boundaries();

        FlipEngine.PressureCompute();

        FlipEngine.AddPressure();

        FlipEngine.TransferToParticule();

        FlipEngine.Integrate();

        FlipEngine.EndCompute();

#endif
        
        //--------------------------------------------------------------------------------

        glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized

        //glDrawElements(GL_TRIANGLES, 200000, GL_UNSIGNED_INT, 0);
        glDrawElementsInstanced(GL_TRIANGLES, vertcount, GL_UNSIGNED_INT, 0, position.size());

        //glDrawArrays(GL_TRIANGLES, 0, 3);
        // glBindVertexArray(0); // no need to unbind it every time 

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    //glDeleteBuffers(1, &EBO);
    //glDeleteProgram(shaderProgram);

    ///7PartEngine.endSystem();

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    SCR_WIDTH = width;
    SCR_HEIGHT = height;

    glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}