
async function api(path, options){ const r = await fetch(path, options); return await r.json(); }
async function startCamera(){ await api('/api/start',{method:'POST'}); refresh(); }
async function stopCamera(){ await api('/api/stop',{method:'POST'}); refresh(); }
async function startScan(){ await api('/api/scan',{method:'POST',body:new URLSearchParams({enabled:'1'})}); refresh(); }
async function stopScan(){ await api('/api/scan',{method:'POST',body:new URLSearchParams({enabled:'0'})}); refresh(); }
async function setMode(mode){ await api('/api/mode',{method:'POST',body:new URLSearchParams({mode})}); refresh(); }
async function selectActiveClass(className){
  await api('/api/active-class',{method:'POST',body:new URLSearchParams({class_name: className})});
  refresh();
}
async function createClass(){
  const className = document.querySelector('#newClassName').value.trim();
  const data = await api('/api/classes',{method:'POST',body:new URLSearchParams({class_name: className})});
  document.querySelector('#registerResult').textContent = data.message;
  refresh();
}
async function registerFace(){
  const name = document.querySelector('#studentName').value.trim();
  const className = document.querySelector('#classSelect').value.trim();
  const data = await api('/api/register',{method:'POST',body:new URLSearchParams({name, class_name: className})});
  document.querySelector('#registerResult').textContent = data.message;
  refresh();
}
async function deleteStudent(name, className){
  await api('/api/student/delete',{method:'POST',body:new URLSearchParams({name, class_name: className})});
  refresh();
}
function toggleClassBlock(id){
  const el = document.getElementById('students_' + id);
  if (el) el.hidden = !el.hidden;
}
function htmlEscape(value){
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function jsString(value){
  return String(value).replace(/\/g, '\\').replace(/'/g, "\'");
}
function domId(value){
  return 'class_' + String(value).replace(/[^a-zA-Z0-9_-]/g, '_');
}
async function refresh(){
  const s = await api('/api/status');
  const r = await api('/api/report');
  const students = await api('/api/students');
  document.querySelector('#mStudents').textContent = students.students.length;
  document.querySelector('#mLen').textContent = r.len_count;
  document.querySelector('#mXuong').textContent = r.xuong_count;
  document.querySelector('#mAnomaly').textContent = r.anomaly_count;
  document.querySelector('#lastResult').textContent = `Mode ${s.mode.toUpperCase()} | Lop ${s.active_class || 'tat ca/chua chon'} | ${s.scan_enabled ? 'dang nhan dien' : 'camera chup mau'} | ${s.last_result.name} | score ${s.last_result.score} | ${s.last_result.message}`;
  document.querySelector('#activeClassResult').textContent = s.active_class ? `Lop dang cho: ${s.active_class}` : 'Chua chon lop dang cho.';
  document.querySelector('#classButtons').innerHTML = s.classes.map(c=>`<button class="btn ${c===s.active_class?'alt':''}" onclick="selectActiveClass('${jsString(c)}')">${htmlEscape(c)}</button>`).join('') || '<span class="sub">Chua co lop. Hay tao lop ben phai.</span>';
  const classSelect = document.querySelector('#classSelect');
  const currentClass = classSelect.value;
  classSelect.innerHTML = s.classes.map(c=>`<option value="${htmlEscape(c)}">${htmlEscape(c)}</option>`).join('') || '<option value="">Chua co lop</option>';
  if (s.classes.includes(currentClass)) classSelect.value = currentClass;
  document.querySelector('#records').innerHTML = r.records.map(x=>`<div class="item"><b>${htmlEscape(x.student_name)}</b> - ${htmlEscape(x.class_name || '')}<br>${x.mode} - ${x.confidence.toFixed(2)} - ${x.created_at}</div>`).join('') || '<div class="item">Chua co nhat ky</div>';
  document.querySelector('#studentList').innerHTML = students.classes.map(cls=>{
    const id = domId(cls.class_name);
    const rows = cls.students.map(x=>`<div class="item"><b>${htmlEscape(x.name)}</b><br>${x.templates} mau | Len ${x.len} / Xuong ${x.xuong}<br><button class="btn danger" onclick="deleteStudent('${jsString(x.name)}', '${jsString(x.class_name)}')">Xoa</button></div>`).join('');
    return `<div class="item"><button class="btn alt" onclick="toggleClassBlock('${id}')">${htmlEscape(cls.class_name)} - ${cls.count} hoc sinh</button><div id="students_${id}" style="margin-top:8px" hidden>${rows || '<div class="item">Lop nay chua co hoc sinh</div>'}</div></div>`;
  }).join('') || '<div class="item">Chua co lop/hoc sinh</div>';
}
let refreshBusy = false;
async function refreshSafe(){
  if (refreshBusy) return;
  refreshBusy = true;
  try {
    await refresh();
  } catch (err) {
    document.querySelector('#lastResult').textContent = 'Loi tai du lieu: ' + err;
  } finally {
    refreshBusy = false;
  }
}
function refreshFrame(){
  document.querySelector('#streamFrame').src = '/frame.jpg?t=' + Date.now();
}
setInterval(()=>{document.querySelector('#clock').textContent = new Date().toLocaleString('vi-VN'); refreshSafe();}, 1500);
setInterval(refreshFrame, 250);
refreshSafe();
