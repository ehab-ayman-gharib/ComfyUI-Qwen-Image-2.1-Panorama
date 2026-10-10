import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * ComfyUI-Qwen-Panorama: Interactive In-Canvas 360° VR Equirectangular Viewer
 * Pure WebGL hardware-accelerated sphere projection with zero external dependencies.
 */

const VERTEX_SHADER_SRC = `
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER_SRC = `
precision highp float;
uniform vec2 u_resolution;
uniform vec2 u_rotation; // x: yaw, y: pitch
uniform float u_fov;
uniform sampler2D u_texture;

#define PI 3.14159265359

void main() {
    // Normalized coordinates centered
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    
    // Ray direction in camera space (uv.y > 0 is up)
    float z = 0.5 / tan(u_fov * 0.5);
    vec3 ray = normalize(vec3(uv.x, uv.y, z));
    
    // Pitch rotation (around X)
    float cp = cos(u_rotation.y);
    float sp = sin(u_rotation.y);
    vec3 r1 = vec3(ray.x, cp * ray.y + sp * ray.z, -sp * ray.y + cp * ray.z);
    
    // Yaw rotation (around Y)
    float cy = cos(u_rotation.x);
    float sy = sin(u_rotation.x);
    vec3 r2 = vec3(cy * r1.x + sy * r1.z, r1.y, -sy * r1.x + cy * r1.z);
    
    // Spherical coordinates -> equirectangular UV
    float lon = atan(r2.x, r2.z);
    float lat = asin(clamp(r2.y, -1.0, 1.0));
    
    vec2 pano_uv = vec2(0.5 + lon / (2.0 * PI), 0.5 - lat / PI);
    
    // Wrap UV seamlessly
    pano_uv.x = fract(pano_uv.x);
    pano_uv.y = clamp(pano_uv.y, 0.001, 0.999);
    
    gl_FragColor = texture2D(u_texture, pano_uv);
}
`;

class WebGLPanoRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext("webgl", { preserveDrawingBuffer: true, alpha: false });
        if (!this.gl) {
            console.error("[QwenPanorama] WebGL not supported");
            return;
        }

        this.initShaders();
        this.initBuffers();
        this.initTexture();

        this.yaw = 0.0;
        this.pitch = 0.0;
        this.fov = 75.0 * (Math.PI / 180.0);
        this.textureLoaded = false;
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };
    }

    initShaders() {
        const gl = this.gl;
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, VERTEX_SHADER_SRC);
        gl.compileShader(vs);

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, FRAGMENT_SHADER_SRC);
        gl.compileShader(fs);

        const program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);

        this.program = program;
        this.uResolution = gl.getUniformLocation(program, "u_resolution");
        this.uRotation = gl.getUniformLocation(program, "u_rotation");
        this.uFov = gl.getUniformLocation(program, "u_fov");
        this.uTexture = gl.getUniformLocation(program, "u_texture");
    }

    initBuffers() {
        const gl = this.gl;
        const quad = new Float32Array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0
        ]);
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

        const aPosition = gl.getAttribLocation(this.program, "a_position");
        gl.enableVertexAttribArray(aPosition);
        gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);
    }

    initTexture() {
        const gl = this.gl;
        this.texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        // Default 1x1 placeholder
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([30, 35, 45, 255]));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    }

    loadTextureFromImage(img) {
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        this.textureLoaded = true;
        this.render();
    }

    render() {
        if (!this.gl || !this.program) return;
        const gl = this.gl;

        if (this.canvas.width !== this.canvas.clientWidth || this.canvas.height !== this.canvas.clientHeight) {
            this.canvas.width = this.canvas.clientWidth || 400;
            this.canvas.height = this.canvas.clientHeight || 240;
        }

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.useProgram(this.program);

        gl.uniform2f(this.uResolution, this.canvas.width, this.canvas.height);
        gl.uniform2f(this.uRotation, this.yaw, this.pitch);
        gl.uniformfloat = gl.uniform1f(this.uFov, this.fov);
        gl.uniform1i(this.uTexture, 0);

        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
}

app.registerExtension({
    name: "ComfyUI.QwenPanorama.Viewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "QwenPanorama360Viewer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.setupPanoramaViewer();
                return r;
            };

            nodeType.prototype.setupPanoramaViewer = function () {
                this.size = [460, 360];

                // Create container element
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#12141a";
                container.style.borderRadius = "8px";
                container.style.overflow = "hidden";
                container.style.border = "1px solid #2d3440";
                container.style.position = "relative";
                container.style.userSelect = "none";

                // Top Header Overlay
                const header = document.createElement("div");
                header.style.position = "absolute";
                header.style.top = "8px";
                header.style.left = "8px";
                header.style.right = "8px";
                header.style.display = "flex";
                header.style.justifyContent = "space-between";
                header.style.alignItems = "center";
                header.style.pointerEvents = "auto";
                header.style.zIndex = "10";

                const badge = document.createElement("span");
                badge.textContent = "🌐 360° VR Panorama";
                badge.style.background = "rgba(15, 23, 42, 0.85)";
                badge.style.color = "#f59e0b";
                badge.style.padding = "3px 8px";
                badge.style.borderRadius = "5px";
                badge.style.fontSize = "11px";
                badge.style.fontWeight = "bold";
                badge.style.border = "1px solid rgba(245, 158, 11, 0.3)";

                const controls = document.createElement("div");
                controls.style.display = "flex";
                controls.style.gap = "5px";

                const btnFullscreen = document.createElement("button");
                btnFullscreen.textContent = "⛶ Full";
                btnFullscreen.style.background = "rgba(15, 23, 42, 0.85)";
                btnFullscreen.style.color = "#e2e8f0";
                btnFullscreen.style.border = "1px solid #334155";
                btnFullscreen.style.padding = "3px 7px";
                btnFullscreen.style.borderRadius = "4px";
                btnFullscreen.style.fontSize = "10px";
                btnFullscreen.style.cursor = "pointer";

                controls.appendChild(btnFullscreen);
                header.appendChild(badge);
                header.appendChild(controls);
                container.appendChild(header);

                // WebGL Canvas
                const canvas = document.createElement("canvas");
                canvas.style.width = "100%";
                canvas.style.height = "100%";
                canvas.style.flex = "1";
                canvas.style.cursor = "grab";
                container.appendChild(canvas);

                // Bottom Hint Bar
                const hintBar = document.createElement("div");
                hintBar.textContent = "Drag to look around (360°/180°) • Scroll to zoom FOV";
                hintBar.style.position = "absolute";
                hintBar.style.bottom = "6px";
                hintBar.style.left = "0";
                hintBar.style.right = "0";
                hintBar.style.textAlign = "center";
                hintBar.style.color = "#94a3b8";
                hintBar.style.fontSize = "10px";
                hintBar.style.pointerEvents = "none";
                container.appendChild(hintBar);

                // Attach WebGL Renderer
                const renderer = new WebGLPanoRenderer(canvas);
                this.panoRenderer = renderer;

                // Responsive resize observer
                const resizeObserver = new ResizeObserver(() => {
                    if (renderer.textureLoaded) {
                        renderer.render();
                    }
                });
                resizeObserver.observe(container);

                // Mouse interaction - strictly only moves when actively dragging with left mouse button
                canvas.addEventListener("pointerdown", (e) => {
                    if (e.button !== 0) return; // Left mouse button only
                    e.stopPropagation();
                    e.preventDefault();
                    renderer.isDragging = true;
                    renderer.lastMouse = { x: e.clientX, y: e.clientY };
                    try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
                    canvas.style.cursor = "grabbing";
                });

                canvas.addEventListener("pointermove", (e) => {
                    if (!renderer.isDragging) return;
                    if (e.buttons !== 1) {
                        // Button released elsewhere or not held down: immediately stop dragging
                        renderer.isDragging = false;
                        canvas.style.cursor = "grab";
                        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
                        return;
                    }
                    e.stopPropagation();
                    e.preventDefault();
                    const dx = e.clientX - renderer.lastMouse.x;
                    const dy = e.clientY - renderer.lastMouse.y;
                    renderer.lastMouse = { x: e.clientX, y: e.clientY };

                    renderer.yaw -= dx * 0.005;
                    renderer.pitch -= dy * 0.005;
                    // Clamp pitch to avoid gimbal flip
                    renderer.pitch = Math.max(-1.45, Math.min(1.45, renderer.pitch));
                    renderer.render();
                });

                const stopDrag = (e) => {
                    if (renderer.isDragging) {
                        renderer.isDragging = false;
                        canvas.style.cursor = "grab";
                        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
                    }
                };

                canvas.addEventListener("pointerup", stopDrag);
                canvas.addEventListener("pointercancel", stopDrag);

                // Zoom (FOV)
                canvas.addEventListener("wheel", (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    const delta = Math.sign(e.deltaY) * 0.08;
                    renderer.fov = Math.max(0.4, Math.min(2.0, renderer.fov + delta));
                    renderer.render();
                }, { passive: false });



                // Fullscreen modal preview
                btnFullscreen.addEventListener("click", () => {
                    if (!document.fullscreenElement) {
                        container.requestFullscreen().catch(err => alert("Fullscreen error: " + err.message));
                    } else {
                        document.exitFullscreen();
                    }
                });

                // Mount widget to LiteGraph node
                const widget = this.addDOMWidget("panorama_viewer", "custom_pano", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                this.onRemoved = function () {
                    resizeObserver.disconnect();
                };
            };

            // Hook executed node to load generated image
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                if (message?.images && message.images.length > 0 && this.panoRenderer) {
                    const imgInfo = message.images[0];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(imgInfo.filename)}&type=${imgInfo.type}&subfolder=${encodeURIComponent(imgInfo.subfolder || "")}&t=${Date.now()}`);

                    const img = new Image();
                    img.crossOrigin = "anonymous";
                    img.onload = () => {
                        this.panoRenderer.loadTextureFromImage(img);
                    };
                    img.src = url;
                }
            };
        }
    }
});
