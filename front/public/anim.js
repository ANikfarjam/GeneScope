import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Lighting
const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(5, 5, 5).normalize();
scene.add(light);

// Chromosome shape
const chromosomeMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, wireframe: false });
const createChromosomePart = (x, y, z) => {
    const geometry = new THREE.CylinderGeometry(0.2, 0.2, 1.5, 32);
    const mesh = new THREE.Mesh(geometry, chromosomeMaterial);
    mesh.position.set(x, y, z);
    mesh.rotation.z = Math.PI / 4;
    return mesh;
};

const leftArm = createChromosomePart(-0.5, 1, 0);
const rightArm = createChromosomePart(0.5, 1, 0);
const leftLeg = createChromosomePart(-0.5, -1, 0);
const rightLeg = createChromosomePart(0.5, -1, 0);
scene.add(leftArm, rightArm, leftLeg, rightLeg);

// DNA Helix (basic structure)
const dnaMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
const dnaHelix = new THREE.Group();
for (let i = 0; i < 10; i++) {
    const dnaSegment = new THREE.TorusGeometry(0.2, 0.05, 16, 100);
    const dnaMesh = new THREE.Mesh(dnaSegment, dnaMaterial);
    dnaMesh.position.set(0, i * 0.3 - 1.5, 0);
    dnaMesh.rotation.x = Math.PI / 2;
    dnaHelix.add(dnaMesh);
}
dnaHelix.position.x = 1.5;
scene.add(dnaHelix);

// Animation
let angle = 0;
function animate() {
    requestAnimationFrame(animate);
    leftArm.rotation.y = Math.sin(angle) * 0.2;
    rightArm.rotation.y = -Math.sin(angle) * 0.2;
    leftLeg.rotation.y = Math.sin(angle) * 0.2;
    rightLeg.rotation.y = -Math.sin(angle) * 0.2;
    dnaHelix.rotation.y += 0.02;
    angle += 0.02;
    renderer.render(scene, camera);
}

animate();

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
camera.position.z = 5;
