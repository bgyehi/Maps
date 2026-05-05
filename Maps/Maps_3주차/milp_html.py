from __future__ import annotations

import html
import json
from pathlib import Path


def schedule_to_visualization_dict(result):
    """
    Convert Schedule result into a JSON-serializable dict
    for standalone HTML visualization.
    """
    inst = result.instance

    jobs_payload = []
    for machine in inst.machine_list:
        assigned = sorted(machine.assigned, key=lambda j: j.start)
        prev_job = None

        for job in assigned:
            if prev_job is None:
                setup_time = 0
            else:
                setup_time = inst.setup[machine.ID][prev_job.ID][job.ID]

            tardiness = max(job.end - job.due, 0)

            jobs_payload.append(
                {
                    "id": int(job.ID),
                    "label": f"Job {int(job.ID) + 1}",
                    "machine": int(machine.ID),
                    "start": float(job.start),
                    "end": float(job.end),
                    "due": float(job.due),
                    "weight": float(job.weight),
                    "setup": float(setup_time),
                    "tardiness": float(tardiness),
                    "weightedTardiness": float(job.weight * tardiness),
                }
            )
            prev_job = job

    payload = {
        "algorithm": str(getattr(result, "algorithm", "MILP_CPLEX")),
        "objective": float(getattr(result, "objective", 0)),
        "status": str(getattr(result, "status", "UNKNOWN")),
        "comp_time": float(getattr(result, "comp_time", 0) or 0),
        "objective_type": str(getattr(inst, "objective", "wT")),
        "machines": [
            {"id": int(m.ID), "name": f"Machine {int(m.ID)}"}
            for m in inst.machine_list
        ],
        "jobs": sorted(
            jobs_payload,
            key=lambda x: (x["machine"], x["start"], x["id"])
        ),
    }
    return payload


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__TITLE__</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    :root {
      --bg: #f6f8fb;
      --card: #ffffff;
      --line: #dde3ec;
      --text: #1f2937;
      --muted: #6b7280;
      --danger: #dc2626;
      --setup: #9ca3af;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }
    .top {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      margin-bottom: 20px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.06);
      padding: 18px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .stat {
      background: #fbfdff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
    }
    .label {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .value {
      font-size: 22px;
      font-weight: 800;
    }
    .legend {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 4px;
      display: inline-block;
      border: 1px solid rgba(0,0,0,0.12);
    }
    .dash {
      width: 18px;
      border-top: 3px dashed var(--setup);
      display: inline-block;
      transform: translateY(-1px);
    }
    .table-wrap {
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    thead th {
      position: sticky;
      top: 0;
      background: #f8fafc;
      z-index: 1;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid #edf2f7;
      text-align: right;
    }
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2) {
      text-align: left;
    }
    svg {
      width: 100%;
      height: 720px;
      display: block;
      background: white;
      border-radius: 18px;
      border: 1px solid var(--line);
      box-shadow: 0 10px 30px rgba(15,23,42,0.06);
    }
    .tooltip {
      position: fixed;
      pointer-events: none;
      opacity: 0;
      background: rgba(15,23,42,0.96);
      color: white;
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.55;
      box-shadow: 0 10px 24px rgba(0,0,0,0.18);
      transform: translate(10px, 10px);
      transition: opacity 0.15s ease;
      z-index: 10;
      max-width: 280px;
    }
    @media (max-width: 980px) {
      .top {
        grid-template-columns: 1fr;
      }
      svg {
        height: 840px;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="card">
        <h1>__TITLE__</h1>
        <p class="sub">
          Python에서 생성한 스케줄 결과를 그대로 HTML로 저장한 파일입니다.
          이 파일은 단독 실행형이라 더블클릭해서 바로 열 수 있습니다.
        </p>

        <div class="stats">
          <div class="stat"><div class="label">Objective</div><div class="value" id="objectiveValue">-</div></div>
          <div class="stat"><div class="label">Status</div><div class="value" id="statusValue">-</div></div>
          <div class="stat"><div class="label">Jobs</div><div class="value" id="jobCountValue">0</div></div>
          <div class="stat"><div class="label">Machines</div><div class="value" id="machineCountValue">0</div></div>
        </div>

        <div class="legend">
          <span><i class="swatch" style="background:#fef3c7"></i> on-time</span>
          <span><i class="swatch" style="background:#fee2e2"></i> tardy</span>
          <span><i class="dash"></i> setup interval</span>
        </div>
      </div>

      <div class="card">
        <h2>Job Summary</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Job</th>
                <th>Machine</th>
                <th>Start</th>
                <th>End</th>
                <th>Due</th>
                <th>Setup</th>
                <th>Tardiness</th>
                <th>Weighted Tardiness</th>
              </tr>
            </thead>
            <tbody id="jobTableBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <svg id="chart"></svg>
    <div id="tooltip" class="tooltip"></div>
  </div>

  <script>
    const data = __DATA_JSON__;
    const svg = d3.select("#chart");
    const tooltip = d3.select("#tooltip");
    const colorScale = d3.scaleOrdinal(d3.schemeTableau10);

    function updateSummary() {
      document.getElementById("objectiveValue").textContent = data.objective ?? "-";
      document.getElementById("statusValue").textContent = data.status ?? "-";
      document.getElementById("jobCountValue").textContent = data.jobs.length;
      document.getElementById("machineCountValue").textContent = data.machines.length;

      const tbody = document.getElementById("jobTableBody");
      tbody.innerHTML = "";

      data.jobs.forEach(job => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${job.label}</td>
          <td>Machine ${job.machine}</td>
          <td>${job.start}</td>
          <td>${job.end}</td>
          <td>${job.due}</td>
          <td>${job.setup}</td>
          <td>${job.tardiness}</td>
          <td>${job.weightedTardiness}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function render() {
      svg.selectAll("*").remove();
      if (!data.jobs.length) return;

      const margin = { top: 70, right: 40, bottom: 80, left: 130 };
      const width = svg.node().clientWidth;
      const height = svg.node().clientHeight;

      const machineNames = new Map(data.machines.map(m => [m.id, m.name]));
      const maxTime = d3.max(data.jobs, d => d.end) || 0;
      const yDomain = data.machines.map(m => m.id);

      const x = d3.scaleLinear()
        .domain([0, maxTime + 10])
        .range([margin.left, width - margin.right]);

      const y = d3.scaleBand()
        .domain(yDomain)
        .range([margin.top, height - margin.bottom])
        .padding(0.24);

      const xAxis = d3.axisBottom(x)
        .ticks(Math.min(15, maxTime + 1))
        .tickFormat(d3.format("d"));

      const yAxis = d3.axisLeft(y)
        .tickFormat(d => machineNames.get(d));

      svg.append("g")
        .attr("transform", `translate(0, ${height - margin.bottom})`)
        .call(xAxis)
        .call(g => g.selectAll("line").attr("stroke", "#d6dce8"))
        .call(g => g.selectAll("path").attr("stroke", "#94a3b8"));

      svg.append("g")
        .attr("transform", `translate(${margin.left}, 0)`)
        .call(yAxis)
        .call(g => g.selectAll("line").remove())
        .call(g => g.selectAll("path").attr("stroke", "#94a3b8"));

      svg.append("text")
        .attr("x", width / 2)
        .attr("y", 34)
        .attr("text-anchor", "middle")
        .attr("font-size", 22)
        .attr("font-weight", 800)
        .attr("fill", "#111827")
        .text(`${data.algorithm} | Total Weighted Tardiness: ${data.objective}`);

      svg.append("text")
        .attr("x", width / 2)
        .attr("y", 56)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#6b7280")
        .text("막대는 processing, 점선은 setup 간격, 빨간 선은 due date");

      svg.append("g")
        .selectAll("line.grid")
        .data(x.ticks(Math.min(20, maxTime + 1)))
        .join("line")
        .attr("x1", d => x(d))
        .attr("x2", d => x(d))
        .attr("y1", margin.top)
        .attr("y2", height - margin.bottom)
        .attr("stroke", "#eef2f7");

      const grouped = d3.group(data.jobs, d => d.machine);
      grouped.forEach((jobs, machineId) => {
        const sorted = [...jobs].sort((a, b) => a.start - b.start);
        sorted.forEach((job, idx) => {
          if (idx === 0) return;
          const prev = sorted[idx - 1];
          if (job.start > prev.end) {
            svg.append("line")
              .attr("x1", x(prev.end))
              .attr("x2", x(job.start))
              .attr("y1", y(machineId) + y.bandwidth() / 2)
              .attr("y2", y(machineId) + y.bandwidth() / 2)
              .attr("stroke", "#9ca3af")
              .attr("stroke-width", 3)
              .attr("stroke-dasharray", "5 4");
          }
        });
      });

      const bars = svg.append("g")
        .selectAll("g.job")
        .data(data.jobs)
        .join("g")
        .attr("class", "job")
        .attr("transform", d => `translate(${x(d.start)}, ${y(d.machine)})`);

      bars.append("rect")
        .attr("width", d => Math.max(4, x(d.end) - x(d.start)))
        .attr("height", y.bandwidth())
        .attr("rx", 10)
        .attr("fill", d => d.tardiness > 0 ? "#fee2e2" : "#fef3c7")
        .attr("stroke", d => d.tardiness > 0 ? "#ef4444" : "#f59e0b")
        .attr("stroke-width", 1.4)
        .on("mousemove", function(event, d) {
          tooltip
            .style("opacity", 1)
            .style("left", event.clientX + "px")
            .style("top", event.clientY + "px")
            .html(`
              <strong>${d.label}</strong><br>
              Machine: ${machineNames.get(d.machine)}<br>
              Start-End: ${d.start} - ${d.end}<br>
              Setup: ${d.setup}<br>
              Due: ${d.due}<br>
              Tardiness: ${d.tardiness}<br>
              Weighted tardiness: ${d.weightedTardiness}
            `);
        })
        .on("mouseleave", () => tooltip.style("opacity", 0));

      bars.append("rect")
        .attr("width", 8)
        .attr("height", y.bandwidth())
        .attr("rx", 10)
        .attr("fill", d => colorScale(d.machine))
        .attr("opacity", 0.95);

      bars.append("text")
        .attr("x", d => (x(d.end) - x(d.start)) / 2)
        .attr("y", y.bandwidth() / 2 + 4)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("font-weight", 800)
        .text(d => d.label.replace("Job ", "J"));

      svg.append("g")
        .selectAll("line.due")
        .data(data.jobs)
        .join("line")
        .attr("x1", d => x(d.due))
        .attr("x2", d => x(d.due))
        .attr("y1", d => y(d.machine) - 8)
        .attr("y2", d => y(d.machine) + y.bandwidth() + 8)
        .attr("stroke", "#dc2626")
        .attr("stroke-width", 1.2)
        .attr("opacity", 0.55);
    }

    updateSummary();
    render();
  </script>
</body>
</html>
"""


def export_schedule_to_html(
    result,
    output_path: str = "solution_result.html",
    title: str = "PMSP-SDST Schedule Result"
) -> Path:
    """
    Create a standalone HTML file that opens by double-clicking
    and immediately shows the solved schedule.
    """
    payload = schedule_to_visualization_dict(result)

    html_text = HTML_TEMPLATE
    html_text = html_text.replace("__TITLE__", html.escape(title))
    html_text = html_text.replace(
        "__DATA_JSON__",
        json.dumps(payload, ensure_ascii=False)
    )

    output = Path(output_path)
    output.write_text(html_text, encoding="utf-8")
    return output