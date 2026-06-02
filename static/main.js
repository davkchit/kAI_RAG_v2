/* ══ kAI landing — main.js ══ */

/* ─── Year ─── */
document.getElementById('year').textContent = new Date().getFullYear();

/* ─── Smooth scroll (Lenis) ─── */
let lenis;
try {
  lenis = new Lenis({ duration: 1.2, easing: t => Math.min(1, 1.001 - Math.pow(2, -10 * t)) });
  gsap.ticker.add(t => lenis.raf(t * 1000));
  gsap.ticker.lagSmoothing(0);
} catch(e) {
  // Lenis unavailable — native scroll
}

/* ─── Navigation ─── */
const nav = document.getElementById('nav');
const navProgress = document.getElementById('navProgress');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 60);
  const pct = window.scrollY / (document.documentElement.scrollHeight - window.innerHeight) * 100;
  navProgress.style.width = Math.min(pct, 100) + '%';
}, { passive: true });

const burger = document.getElementById('burger');
const mobileMenu = document.getElementById('mobileMenu');
burger.addEventListener('click', () => {
  const open = mobileMenu.classList.toggle('open');
  burger.classList.toggle('open', open);
});
mobileMenu.querySelectorAll('a').forEach(a => {
  a.addEventListener('click', () => {
    mobileMenu.classList.remove('open');
    burger.classList.remove('open');
  });
});

/* ─── Custom cursor ─── */
const ring = document.getElementById('cursor-ring');
const dot  = document.getElementById('cursor-dot');
let mx = -100, my = -100, rx = -100, ry = -100;

if (window.matchMedia('(pointer:fine)').matches) {
  document.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });

  (function animCursor() {
    const ease = 0.14;
    rx += (mx - rx) * ease;
    ry += (my - ry) * ease;
    ring.style.left = rx + 'px';
    ring.style.top  = ry + 'px';
    dot.style.left  = mx + 'px';
    dot.style.top   = my + 'px';
    requestAnimationFrame(animCursor);
  })();

  document.addEventListener('mouseover', e => {
    if (e.target.closest('a, button, [data-cursor-expand]'))
      ring.classList.add('expanded');
  });
  document.addEventListener('mouseout', e => {
    if (e.target.closest('a, button, [data-cursor-expand]'))
      ring.classList.remove('expanded');
  });
}

/* ─── Fade-in on scroll (IntersectionObserver) ─── */
const io = new IntersectionObserver(entries => {
  entries.forEach((e, idx) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (parseFloat(e.target.dataset.d || 0) * 0.1) + 's';
      e.target.classList.add('vis');
      io.unobserve(e.target);
    }
  });
}, { threshold: 0.12, rootMargin: '-40px' });

document.querySelectorAll('.fi').forEach((el, i) => {
  // stagger siblings: find index among .fi siblings
  const siblings = Array.from(el.parentElement.querySelectorAll('.fi'));
  const idx = siblings.indexOf(el);
  if (idx > 0) el.dataset.d = String(idx);
  io.observe(el);
});

/* ─── Statement — GSAP ScrollTrigger ─── */
gsap.registerPlugin(ScrollTrigger);

const stWords   = [0,1,2,3].map(i => document.getElementById('stWord' + i));
const stFinal   = document.getElementById('stFinal');
const statement = document.getElementById('statement');

const tl = gsap.timeline({
  scrollTrigger: {
    trigger: statement,
    start: 'top top',
    end: '+=450%',
    pin: true,
    scrub: 0.9,
    anticipatePin: 1,
  }
});

stWords.forEach((el, i) => {
  const offset = i * 0.22;
  tl.fromTo(el, { opacity: 0, y: 50 }, { opacity: 1, y: 0, duration: 0.18, ease: 'none' }, offset);
  if (i < stWords.length - 1) {
    tl.to(el, { opacity: 0, y: -40, duration: 0.14, ease: 'none' }, offset + 0.16);
  } else {
    tl.to(el, { opacity: 0, y: -40, duration: 0.14, ease: 'none' }, offset + 0.2);
  }
});

tl.fromTo(stFinal, { opacity: 0, y: 32 }, { opacity: 1, y: 0, duration: 0.22, ease: 'none' }, stWords.length * 0.22);

if (lenis) {
  lenis.on('scroll', ScrollTrigger.update);
}

/* ─── Programs 3D tilt ─── */
document.querySelectorAll('.prog-card').forEach(card => {
  let raf;
  let tx = 0, ty = 0, cx = 0, cy = 0;

  card.addEventListener('mousemove', e => {
    const r = card.getBoundingClientRect();
    tx = (e.clientX - r.left) / r.width  - 0.5;
    ty = (e.clientY - r.top)  / r.height - 0.5;
  });

  card.addEventListener('mouseenter', () => {
    card.style.transition = 'none';
    (function tick() {
      cx += (tx - cx) * 0.12;
      cy += (ty - cy) * 0.12;
      card.style.transform = `perspective(800px) rotateX(${-cy * 5}deg) rotateY(${cx * 5}deg)`;
      raf = requestAnimationFrame(tick);
    })();
  });

  card.addEventListener('mouseleave', () => {
    cancelAnimationFrame(raf);
    card.style.transition = 'transform 0.6s cubic-bezier(0.16,1,0.3,1)';
    card.style.transform = 'perspective(800px) rotateX(0) rotateY(0)';
    tx = ty = cx = cy = 0;
  });
});

/* ─── Three.js sphere ─── */
(function initSphere() {
  const canvas = document.getElementById('sphereCanvas');
  if (!canvas || !window.THREE) return;
  if (window.matchMedia('(max-width:767px)').matches) return;

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
  camera.position.z = 6;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);

  // Wireframe sphere
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(2.2, 24, 24),
    new THREE.MeshBasicMaterial({ color: 0xA0C4FF, wireframe: true, transparent: true, opacity: 0.07 })
  );
  scene.add(sphere);

  // Neural points (Fibonacci lattice on sphere)
  const count = 600;
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const phi   = Math.acos(1 - (2 * (i + 0.5)) / count);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    pos[i*3]   = Math.sin(phi) * Math.cos(theta) * 2.2;
    pos[i*3+1] = Math.sin(phi) * Math.sin(theta) * 2.2;
    pos[i*3+2] = Math.cos(phi) * 2.2;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const points = new THREE.Points(geo,
    new THREE.PointsMaterial({ color: 0xC8DCFF, size: 0.028, sizeAttenuation: true, transparent: true, opacity: 0.75, depthWrite: false })
  );
  scene.add(points);

  // Pulse ring
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xFFFFFF, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.22, 32), ringMat);
  scene.add(ring);

  new ResizeObserver(() => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }).observe(canvas);

  const t0 = performance.now();
  (function animate() {
    requestAnimationFrame(animate);
    const t = (performance.now() - t0) / 1000;
    sphere.rotation.y  =  t * -0.05;
    sphere.rotation.z  =  t *  0.03;
    points.rotation.y  =  t *  0.08;
    points.rotation.x  = Math.sin(t * 0.04) * 0.12;
    const pt = (t * 0.4) % 1;
    ring.scale.setScalar(1 + pt * 1.8);
    ringMat.opacity = (1 - pt) * 0.3;
    renderer.render(scene, camera);
  })();
})();

/* ─── Cosmos canvas (Finale) ─── */
(function initCosmos() {
  const canvas = document.getElementById('cosmosCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const resize = () => {
    canvas.width  = canvas.offsetWidth  * devicePixelRatio;
    canvas.height = canvas.offsetHeight * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  };
  resize();
  window.addEventListener('resize', resize, { passive: true });

  const stars = Array.from({ length: 320 }, () => ({
    x: Math.random(), y: Math.random(),
    r: Math.pow(Math.random(), 2.2) * 1.1 + 0.14,
    baseO: Math.random() * 0.5 + 0.18,
    speed: Math.random() * 0.0005 + 0.00015,
    phase: Math.random() * Math.PI * 2,
  }));

  const bright = Array.from({ length: 6 }, () => ({
    x: Math.random(), y: Math.random(),
    r: Math.random() * 1.0 + 1.0,
    o: 0.6 + Math.random() * 0.3,
  }));

  (function draw(t) {
    requestAnimationFrame(draw);
    const w = canvas.offsetWidth, h = canvas.offsetHeight;
    ctx.clearRect(0, 0, w, h);

    for (const s of stars) {
      ctx.beginPath();
      ctx.arc(s.x*w, s.y*h, s.r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(205,220,255,${s.baseO*(0.68+0.32*Math.sin(t*s.speed+s.phase))})`;
      ctx.fill();
    }

    for (const s of bright) {
      const sx = s.x*w, sy = s.y*h;
      const g = ctx.createRadialGradient(sx,sy,0,sx,sy,s.r*11);
      g.addColorStop(0, `rgba(160,200,255,${s.o*0.45})`);
      g.addColorStop(1, 'rgba(160,200,255,0)');
      ctx.beginPath(); ctx.arc(sx,sy,s.r*11,0,Math.PI*2); ctx.fillStyle=g; ctx.fill();
      ctx.beginPath(); ctx.arc(sx,sy,s.r,0,Math.PI*2); ctx.fillStyle=`rgba(238,245,255,${s.o})`; ctx.fill();
      const len = s.r*16;
      ctx.save(); ctx.strokeStyle=`rgba(190,215,255,${s.o*0.32})`; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.moveTo(sx-len,sy); ctx.lineTo(sx+len,sy); ctx.moveTo(sx,sy-len); ctx.lineTo(sx,sy+len); ctx.stroke(); ctx.restore();
    }
  })(0);
})();

/* ─── Chat widget ─── */
(function initChat() {
  const widget  = document.getElementById('chatWidget');
  const panel   = document.getElementById('chatPanel');
  const toggle  = document.getElementById('chatToggle');
  const closeBtn= document.getElementById('chatClose');
  const msgs    = document.getElementById('chatMsgs');
  const empty   = document.getElementById('chatEmpty');
  const input   = document.getElementById('chatInput');
  const sendBtn = document.getElementById('chatSend');

  let open = false, loading = false;

  function openChat(v) {
    open = v;
    panel.classList.toggle('open', v);
    widget.classList.toggle('chat-open', v);
    if (v) setTimeout(() => input.focus(), 160);
  }

  toggle.addEventListener('click', () => openChat(!open));
  closeBtn.addEventListener('click', () => openChat(false));

  function scrollBottom() {
    msgs.scrollTop = msgs.scrollHeight;
  }

  function addMsg(role, text) {
    empty.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'msg ' + role;
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    row.appendChild(bubble);
    msgs.appendChild(row);
    scrollBottom();
    return row;
  }

  function addTyping() {
    empty.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'msg bot';
    row.innerHTML = '<div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
    msgs.appendChild(row);
    scrollBottom();
    return row;
  }

  function cleanAnswer(text) {
    return text.replace(/\[src:[^\]]+\]/g, '').trim();
  }

  async function send() {
    const q = input.value.trim();
    if (!q || loading) return;

    loading = true;
    sendBtn.disabled = true;
    addMsg('user', q);
    input.value = '';
    input.style.height = 'auto';

    const typingRow = addTyping();

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      typingRow.remove();
      addMsg('bot', cleanAnswer(data.answer));
    } catch {
      typingRow.remove();
      addMsg('bot', 'Не удалось получить ответ — попробуй ещё раз.');
    } finally {
      loading = false;
      sendBtn.disabled = false;
    }
  }

  sendBtn.addEventListener('click', send);

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  input.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 96) + 'px';
  });
})();
