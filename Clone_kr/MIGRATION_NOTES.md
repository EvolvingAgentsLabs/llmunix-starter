# LLMunix 스타터 템플릿 - 클린 버전 노트

## 개요

이 저장소는 llmunix-marketplace 플러그인 철학에서 영감을 받은 LLMunix의 **최소한의 클린 버전**입니다. 이것은 비대함 없이 필수 커널 구성요소만 제공합니다.

## 철학: 제품이 아닌 공장

전통적인 접근 방식: 다양한 도메인을 위한 많은 사전 구축된 에이전트를 제공합니다.

**LLMunix 접근 방식**: 요청 시 에이전트를 생성하는 최소한의 커널을 제공합니다.

## 포함된 것 (최소 코어)

### 핵심 시스템 에이전트 (3개만)
- **SystemAgent.md** - 마스터 오케스트레이터
- **MemoryAnalysisAgent.md** - 로그 분석기
- **MemoryConsolidationAgent.md** - 학습 통합기

### 시스템 파일
- **SmartMemory.md** - 메모리 아키텍처 명세
- **Tools/** - 쿼리, 추적, 도구 매핑 정의

### 커널
- **CLAUDE.md** - SystemAgent를 구현하는 자동 로딩 커널

### 구조
- **projects/** - 빈 디렉토리 (동적으로 채워짐)
- **.claude/agents/** - 3개의 핵심 에이전트만
- **system/** - 참조용 소스 정의

## 포함되지 않은 것 (의도적으로)

### ❌ 도메인 특화 에이전트 없음
`system/agents/`와 `.claude/agents/` 모두에서 제거됨:
- VisionaryAgent.md
- MathematicianAgent.md
- QuantumEngineerAgent.md

**왜?** 이것들은 특정 프로젝트 요구사항에 따라 동적으로 생성되어야 합니다. 사전 제공하면:
1. 사용자가 특정 문제 해결 패턴으로 편향됨
2. 더 많은 도메인이 추가됨에 따라 비대해짐
3. 프로젝트 요구에 대한 진정한 커스터마이징을 방해함
4. "공장" 철학에 어긋남

### ❌ 예제 프로젝트 없음
전체 `projects/` 디렉토리 내용 제거됨:
- Project_aorta
- Project_chaos_bifurcation_tutorial_v2
- Project_seismic_surveying

**왜?**
1. 사용자가 자신의 프로젝트를 동적으로 생성해야 함
2. 예제가 잘못된 기대를 만듦
3. 저장소 공간을 차지함
4. 사용자가 직접 해봐야 더 잘 배움

### ❌ 대체 런타임 없음
제거됨:
- qwen_runtime.py
- QWEN.md
- GEMINI.md

**왜?** 이것은 웹 최적화 버전입니다. 대체 런타임은 별도로 또는 다른 브랜치에서 문서화될 수 있습니다. 초점은 Claude Code 웹입니다.

### ❌ 시나리오 디렉토리 없음
`scenarios/` 전체 제거됨.

**왜?** 사용 패턴은 미리 정해진 시나리오가 아닌 실제 사용에서 나타나야 합니다.

## 비교: 정리 전 vs 후

### 정리 전 (비대함)
```
.claude/agents/
├── SystemAgent.md                    ✓ 핵심
├── MemoryAnalysisAgent.md            ✓ 핵심
├── MemoryConsolidationAgent.md       ✓ 핵심
├── VisionaryAgent.md                 ✗ 도메인 특화
├── MathematicianAgent.md             ✗ 도메인 특화
├── QuantumEngineerAgent.md           ✗ 도메인 특화
└── Project_*_*.md                    ✗ 프로젝트 특화

system/agents/
├── SystemAgent.md                    ✓ 핵심
├── MemoryAnalysisAgent.md            ✓ 핵심
├── MemoryConsolidationAgent.md       ✓ 핵심
├── VisionaryAgent.md                 ✗ 도메인 특화
├── MathematicianAgent.md             ✗ 도메인 특화
└── QuantumEngineerAgent.md           ✗ 도메인 특화

projects/
├── Project_aorta/                    ✗ 예제
├── Project_chaos_bifurcation.../     ✗ 예제
└── Project_seismic_surveying/        ✗ 예제

루트:
├── qwen_runtime.py                   ✗ 대체 런타임
├── QWEN.md                           ✗ 대체 런타임
├── GEMINI.md                         ✗ 대체 런타임
└── scenarios/                        ✗ 규범적 예제
```

### 정리 후 (클린)
```
.claude/agents/
├── SystemAgent.md                    ✓ 핵심만
├── MemoryAnalysisAgent.md            ✓ 핵심만
└── MemoryConsolidationAgent.md       ✓ 핵심만

system/agents/
├── SystemAgent.md                    ✓ 핵심만
├── MemoryAnalysisAgent.md            ✓ 핵심만
└── MemoryConsolidationAgent.md       ✓ 핵심만

projects/
└── .gitkeep                          ✓ 비어 있음, 동적 생성 준비됨

루트: 필수 요소만
```

## 문서에 대한 주요 변경사항

### CLAUDE.md
- 3개의 핵심 에이전트만 존재한다는 것을 명확히 하도록 **업데이트됨**
- 도메인 에이전트의 오해 소지가 있는 목록 **제거됨**
- 동적 에이전트 생성을 주요 워크플로우로 **강조함**

### README.md
- "제품이 아닌 공장" 철학을 강조하도록 **재작성됨**
- 플러그인과의 명확한 비교 **추가됨**
- 예제 프로젝트 및 대체 런타임에 대한 참조 **제거됨**
- 포함되지 않은 것(기능으로서)을 **강조함**

## 철학 정렬

이 정리된 템플릿은 이제 llmunix-marketplace 플러그인과 완벽하게 일치합니다:

| 구성요소 | 플러그인 | 스타터 (정리 후) |
|-----------|--------|------------------------|
| 핵심 에이전트 | 3 | 3 |
| 도메인 에이전트 | 0 (동적 생성) | 0 (동적 생성) |
| 예제 프로젝트 | 0 | 0 |
| 시스템 파일 | 4 | 4 |
| 철학 | 최소 커널 | 최소 커널 |

## 클린 접근 방식의 이점

### 1. **진정한 동적 생성**
사용자가 제한된 메뉴에서 선택하는 것이 아니라 어떤 에이전트가 필요한지 생각해야 합니다.

### 2. **저장소 크기 감소**
51개 파일(22,665줄)에서 ~15개 핵심 파일로.

### 3. **더 명확한 가치 제안**
"이것은 공장이다"가 구조에서 즉시 명백합니다.

### 4. **더 나은 학습**
사용자가 예제를 복사하는 것이 아니라 자신의 프로젝트를 만들어 학습합니다.

### 5. **더 쉬운 유지보수**
구식이 될 수 있는 예제 프로젝트나 도메인 에이전트를 유지할 필요가 없습니다.

### 6. **확장성**
시스템이 모든 가능성에 대해 에이전트를 사전 제공하지 않고도 모든 도메인을 처리할 수 있습니다.

## 마이그레이션 요약

**제거됨:**
- system/agents/에서 3개의 도메인 특화 에이전트
- .claude/agents/에서 3개의 도메인 특화 에이전트
- .claude/agents/에서 3개의 프로젝트 특화 에이전트
- 모든 내용이 포함된 3개의 예제 프로젝트
- 3개의 대체 런타임 파일
- 1개의 시나리오 디렉토리

**유지됨:**
- 3개의 핵심 시스템 에이전트
- 4개의 시스템 명세 파일
- 1개의 커널 (CLAUDE.md)
- 필수 문서
- 라이선스 및 구성 파일

**결과:** "제품이 아닌 공장" 철학을 구현하는 최소한의 클린 템플릿.

## 정리 후 사용법

1. 이 템플릿을 **포크/클론**
2. Claude Code 웹에 **연결**
3. **목표 제공** - 복잡하고 다면적인 문제
4. Claude가 필요한 정확한 에이전트를 동적으로 생성하는 것을 **관찰**
5. 생성된 에이전트 정의에서 **학습**
6. 메모리 시스템을 통해 향후 프로젝트에서 학습 내용 **재사용**

예제가 필요 없습니다. 사전 구축된 솔루션이 필요 없습니다. 순수하고 동적인 에이전트 생성만 있습니다.

---

**이것이 LLMunix의 본질입니다: 당신의 문제를 해결하기 위해 스스로를 구축하는 자기 진화 운영체제.**
