// Geometric Mnemic Manifolds - Interactive Visualizations
// Using Three.js for 3D rendering

// Global state
const scenes = {};
let memoryCount = 50;
let isRotating = true;

// Utility: Generate Kronecker sequence coordinates
function kroneckerCoordinate(k, dimension, primes) {
    const coords = [];
    for (let i = 0; i < dimension; i++) {
        const alpha = Math.sqrt(primes[i]);
        const frac = (k * alpha) % 1;
        // Inverse error function approximation (simplified)
        const erfInv = approximateErfInv(2 * frac - 1);
        coords.push(erfInv);
    }
    return normalize(coords);
}

function approximateErfInv(x) {
    // Simplified approximation of inverse error function
    const a = 0.147;
    const b = 2 / (Math.PI * a) + Math.log(1 - x * x) / 2;
    const sign = x < 0 ? -1 : 1;
    return sign * Math.sqrt(Math.sqrt(b * b - Math.log(1 - x * x) / a) - b);
}

function normalize(coords) {
    const magnitude = Math.sqrt(coords.reduce((sum, c) => sum + c * c, 0));
    return coords.map(c => c / magnitude);
}

// Utility: Radial decay function
function radialDecay(k, gamma = 1.0) {
    return Math.pow(1 + k, -gamma);
}

// Utility: Color interpolation
function interpolateColor(t, color1, color2) {
    return {
        r: color1.r + (color2.r - color1.r) * t,
        g: color1.g + (color2.g - color1.g) * t,
        b: color1.b + (color2.b - color1.b) * t
    };
}

function colorToHex(color) {
    const r = Math.floor(color.r * 255);
    const g = Math.floor(color.g * 255);
    const b = Math.floor(color.b * 255);
    return (r << 16) | (g << 8) | b;
}

// ============================================================================
// VISUALIZATION 1: Kronecker Spiral on Sphere
// ============================================================================

function initSpiralViz() {
    const container = document.getElementById('spiral-viz');
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    camera.position.z = 3;

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    // Add point light
    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);

    // Add transparent sphere
    const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
    const sphereMaterial = new THREE.MeshBasicMaterial({
        color: 0x667eea,
        wireframe: true,
        transparent: true,
        opacity: 0.1
    });
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    scene.add(sphere);

    // Memory points group
    const memoryGroup = new THREE.Group();
    scene.add(memoryGroup);

    // Generate initial memories
    const primes = [2, 3, 5, 7, 11];
    generateMemories(memoryGroup, memoryCount, primes);

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);

        if (isRotating) {
            memoryGroup.rotation.y += 0.003;
            sphere.rotation.y += 0.003;
        }

        renderer.render(scene, camera);
    }
    animate();

    // Handle window resize
    window.addEventListener('resize', () => {
        if (container.clientWidth > 0) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });

    scenes.spiral = { scene, camera, renderer, memoryGroup, primes };
}

function generateMemories(group, count, primes) {
    // Clear existing memories
    while (group.children.length > 0) {
        group.remove(group.children[0]);
    }

    // Color gradient
    const color1 = { r: 0.4, g: 0.49, b: 0.92 }; // #667eea (recent)
    const color2 = { r: 0.94, g: 0.58, b: 0.98 }; // #f093fb (mid)
    const color3 = { r: 0.31, g: 0.67, b: 0.99 }; // #4facfe (ancient)

    for (let k = 0; k < count; k++) {
        const coords = kroneckerCoordinate(k, 3, primes);
        const radius = 1.0;

        // Convert to 3D position
        const x = coords[0] * radius;
        const y = coords[1] * radius;
        const z = coords[2] * radius;

        // Size based on recency
        const size = 0.02 + 0.03 * (1 - k / count);

        // Color based on age
        let color;
        if (k < count / 3) {
            const t = k / (count / 3);
            color = interpolateColor(t, color1, color2);
        } else {
            const t = (k - count / 3) / (2 * count / 3);
            color = interpolateColor(t, color2, color3);
        }

        const geometry = new THREE.SphereGeometry(size, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: colorToHex(color),
            emissive: colorToHex(color),
            emissiveIntensity: 0.3,
            shininess: 30
        });

        const memory = new THREE.Mesh(geometry, material);
        memory.position.set(x, y, z);
        group.add(memory);
    }
}

// ============================================================================
// VISUALIZATION 2: Temporal Decay
// ============================================================================

function initDecayViz() {
    const container = document.getElementById('decay-viz');
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    camera.position.z = 2.5;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(3, 3, 3);
    scene.add(pointLight);

    // Memory layers group
    const layersGroup = new THREE.Group();
    scene.add(layersGroup);

    // Generate temporal layers
    const primes = [2, 3, 5, 7, 11];
    const numMemories = 100;
    const gamma = 1.0;

    for (let k = 0; k < numMemories; k++) {
        const coords = kroneckerCoordinate(k, 3, primes);
        const r = radialDecay(k, gamma);

        // Position on sphere with radial decay
        const x = coords[0] * r;
        const y = coords[1] * r;
        const z = coords[2] * r;

        // Size decreases with age
        const size = 0.01 + 0.02 * r;

        // Color based on radius (recency)
        let color;
        if (r > 0.6) {
            color = { r: 1.0, g: 0.42, b: 0.42 }; // #ff6b6b (recent)
        } else if (r > 0.3) {
            color = { r: 1.0, g: 0.85, b: 0.24 }; // #ffd93d (medium)
        } else {
            color = { r: 0.29, g: 0.56, b: 0.89 }; // #4a90e2 (ancient)
        }

        const geometry = new THREE.SphereGeometry(size, 12, 12);
        const material = new THREE.MeshPhongMaterial({
            color: colorToHex(color),
            emissive: colorToHex(color),
            emissiveIntensity: 0.4,
            transparent: true,
            opacity: 0.6 + 0.4 * r
        });

        const memory = new THREE.Mesh(geometry, material);
        memory.position.set(x, y, z);
        memory.userData = { originalR: r, k: k };
        layersGroup.add(memory);
    }

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        layersGroup.rotation.y += 0.002;
        renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    window.addEventListener('resize', () => {
        if (container.clientWidth > 0) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });

    scenes.decay = { scene, camera, renderer, layersGroup, primes };
}

// ============================================================================
// VISUALIZATION 3: Hierarchical Layers
// ============================================================================

function initHierarchyViz() {
    const container = document.getElementById('hierarchy-viz');
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    camera.position.set(0, 2, 4);
    camera.lookAt(0, 0, 0);

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    // Three layer groups
    const l0Group = new THREE.Group();
    const l1Group = new THREE.Group();
    const l2Group = new THREE.Group();

    scene.add(l0Group);
    scene.add(l1Group);
    scene.add(l2Group);

    const primes = [2, 3, 5, 7, 11];

    // L0: Raw episodes (100 small dots)
    for (let i = 0; i < 100; i++) {
        const coords = kroneckerCoordinate(i, 3, primes);
        const r = 1.5;
        const geometry = new THREE.SphereGeometry(0.02, 8, 8);
        const material = new THREE.MeshPhongMaterial({
            color: 0x00ffff,
            emissive: 0x00ffff,
            emissiveIntensity: 0.3
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(coords[0] * r, coords[1] * r - 1.5, coords[2] * r);
        l0Group.add(sphere);
    }

    // L1: Pattern nodes (25 medium dots)
    for (let i = 0; i < 25; i++) {
        const coords = kroneckerCoordinate(i * 4, 3, primes);
        const r = 1.3;
        const geometry = new THREE.SphereGeometry(0.04, 12, 12);
        const material = new THREE.MeshPhongMaterial({
            color: 0xffff00,
            emissive: 0xffff00,
            emissiveIntensity: 0.4
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(coords[0] * r, coords[1] * r, coords[2] * r);
        l1Group.add(sphere);
    }

    // L2: Axiom nodes (8 large dots)
    for (let i = 0; i < 8; i++) {
        const coords = kroneckerCoordinate(i * 12, 3, primes);
        const r = 1.1;
        const geometry = new THREE.SphereGeometry(0.06, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: 0xff00ff,
            emissive: 0xff00ff,
            emissiveIntensity: 0.5
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(coords[0] * r, coords[1] * r + 1.5, coords[2] * r);
        l2Group.add(sphere);
    }

    // Animation
    function animate() {
        requestAnimationFrame(animate);
        l0Group.rotation.y += 0.002;
        l1Group.rotation.y += 0.002;
        l2Group.rotation.y += 0.002;
        renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    window.addEventListener('resize', () => {
        if (container.clientWidth > 0) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });

    scenes.hierarchy = { scene, camera, renderer, l0Group, l1Group, l2Group };
}

// ============================================================================
// VISUALIZATION 4: Foveated Attention
// ============================================================================

function initFoveaViz() {
    const container = document.getElementById('fovea-viz');
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    camera.position.set(0, 0, 3);

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1, 100);
    pointLight.position.set(3, 3, 3);
    scene.add(pointLight);

    // Agent at center
    const agentGeometry = new THREE.SphereGeometry(0.1, 32, 32);
    const agentMaterial = new THREE.MeshPhongMaterial({
        color: 0xff0000,
        emissive: 0xff0000,
        emissiveIntensity: 0.6
    });
    const agent = new THREE.Mesh(agentGeometry, agentMaterial);
    scene.add(agent);

    // Memory points
    const memoryGroup = new THREE.Group();
    scene.add(memoryGroup);

    const primes = [2, 3, 5, 7, 11];
    const numMemories = 80;
    const gamma = 0.8;

    for (let k = 0; k < numMemories; k++) {
        const coords = kroneckerCoordinate(k, 3, primes);
        const r = radialDecay(k, gamma) * 2;

        const x = coords[0] * r;
        const y = coords[1] * r;
        const z = coords[2] * r;

        // Classify into fovea regions
        let color, size, opacity;
        if (k < 10) {
            // Foveal
            color = 0x00ff00;
            size = 0.05;
            opacity = 1.0;
        } else if (k < 64) {
            // Para-foveal
            color = 0xffff00;
            size = 0.03;
            opacity = 0.7;
        } else {
            // Peripheral
            color = 0x4444ff;
            size = 0.02;
            opacity = 0.4;
        }

        const geometry = new THREE.SphereGeometry(size, 12, 12);
        const material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: opacity
        });

        const memory = new THREE.Mesh(geometry, material);
        memory.position.set(x, y, z);
        memory.userData = { region: k < 10 ? 'fovea' : k < 64 ? 'parafovea' : 'periphery' };
        memoryGroup.add(memory);

        // Draw connection line to agent
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(x, y, z)
        ]);
        const lineMaterial = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: opacity * 0.3
        });
        const line = new THREE.Line(lineGeometry, lineMaterial);
        memoryGroup.add(line);
    }

    // Animation
    function animate() {
        requestAnimationFrame(animate);
        memoryGroup.rotation.y += 0.002;
        renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    window.addEventListener('resize', () => {
        if (container.clientWidth > 0) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });

    scenes.fovea = { scene, camera, renderer, memoryGroup, agent };
}

// ============================================================================
// Initialize all visualizations
// ============================================================================

function initAllVisualizations() {
    initSpiralViz();
    initDecayViz();
    initHierarchyViz();
    initFoveaViz();
}

// Wait for DOM and Three.js to load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAllVisualizations);
} else {
    initAllVisualizations();
}

// ============================================================================
// Interactive Controls
// ============================================================================

function addMemories() {
    memoryCount += 20;
    if (scenes.spiral && scenes.spiral.memoryGroup) {
        generateMemories(scenes.spiral.memoryGroup, memoryCount, scenes.spiral.primes);
    }
}

function resetSpiral() {
    memoryCount = 50;
    if (scenes.spiral && scenes.spiral.memoryGroup) {
        generateMemories(scenes.spiral.memoryGroup, memoryCount, scenes.spiral.primes);
    }
}

function toggleRotation() {
    isRotating = !isRotating;
}

function animateTime() {
    if (!scenes.decay || !scenes.decay.layersGroup) return;

    const layersGroup = scenes.decay.layersGroup;
    let t = 0;

    const interval = setInterval(() => {
        t += 0.01;

        layersGroup.children.forEach((child) => {
            if (child.userData.k !== undefined) {
                const k = child.userData.k;
                const newK = k + t * 10;
                const newR = radialDecay(newK, 1.0);

                const coords = kroneckerCoordinate(k, 3, scenes.decay.primes);
                child.position.set(
                    coords[0] * newR,
                    coords[1] * newR,
                    coords[2] * newR
                );

                child.material.opacity = 0.6 + 0.4 * newR;
            }
        });

        if (t >= 1.0) clearInterval(interval);
    }, 50);
}

function showDecayFormula() {
    alert('Radial Decay Formula:\n\nr_k = (1 + k)^(-γ)\n\nwhere:\n- k is the temporal index\n- γ controls decay rate\n- Result: memories drift inward over time');
}

function showLayer(layer) {
    if (!scenes.hierarchy) return;

    scenes.hierarchy.l0Group.visible = (layer === 0);
    scenes.hierarchy.l1Group.visible = (layer === 1);
    scenes.hierarchy.l2Group.visible = (layer === 2);
}

function showAllLayers() {
    if (!scenes.hierarchy) return;

    scenes.hierarchy.l0Group.visible = true;
    scenes.hierarchy.l1Group.visible = true;
    scenes.hierarchy.l2Group.visible = true;
}

function highlightFovea() {
    highlightRegion('fovea');
}

function highlightParafovea() {
    highlightRegion('parafovea');
}

function highlightPeriphery() {
    highlightRegion('periphery');
}

function highlightRegion(region) {
    if (!scenes.fovea || !scenes.fovea.memoryGroup) return;

    scenes.fovea.memoryGroup.children.forEach((child) => {
        if (child.type === 'Mesh' && child.userData.region) {
            if (child.userData.region === region) {
                child.material.emissiveIntensity = 0.8;
                child.material.opacity = 1.0;
            } else {
                child.material.emissiveIntensity = 0.1;
                child.material.opacity = 0.2;
            }
        } else if (child.type === 'Line' && child.material) {
            child.material.opacity = 0.1;
        }
    });

    // Reset after 2 seconds
    setTimeout(() => {
        scenes.fovea.memoryGroup.children.forEach((child) => {
            if (child.type === 'Mesh' && child.userData.region) {
                const region = child.userData.region;
                child.material.emissiveIntensity = 0.3;
                child.material.opacity = region === 'fovea' ? 1.0 : region === 'parafovea' ? 0.7 : 0.4;
            } else if (child.type === 'Line' && child.material) {
                const baseOpacity = child.material.color.getHex() === 0x00ff00 ? 0.3 :
                                   child.material.color.getHex() === 0xffff00 ? 0.21 : 0.12;
                child.material.opacity = baseOpacity;
            }
        });
    }, 2000);
}

function animateQuery() {
    if (!scenes.fovea) return;

    // Pulse the agent
    let scale = 1.0;
    let growing = true;
    let count = 0;

    const interval = setInterval(() => {
        if (growing) {
            scale += 0.05;
            if (scale >= 1.5) growing = false;
        } else {
            scale -= 0.05;
            if (scale <= 1.0) {
                growing = true;
                count++;
            }
        }

        scenes.fovea.agent.scale.set(scale, scale, scale);

        if (count >= 3) clearInterval(interval);
    }, 50);
}
