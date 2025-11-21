# LLMunix 스타터: 순수 마크다운 운영체제 템플릿

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**LLMunix**는 사전 구축된 솔루션을 제공하지 않는 순수 마크다운 운영체제입니다—이것은 솔루션을 **만들기 위한 공장**을 제공합니다. 이 스타터 템플릿은 웹의 Claude Code에 최적화되어 있으며, 동적 에이전트 생성과 자기 진화하는 문제 해결을 가능하게 합니다.

## 철학: 제품이 아닌 공장

전통적인 AI 시스템은 특정 도메인을 위한 에이전트를 사전 정의합니다 (예: "파이썬 개발자", "데이터 과학자"). 이것은 근본적인 한계가 있습니다:

❌ **도메인 범위가 제한됨** - 새로운 조합을 처리할 수 없음
❌ **에이전트가 일반적임** - 특정 문제의 뉘앙스에 맞춤화되지 않음
❌ **시스템이 비대해짐** - 수백 개의 사전 구축된 에이전트를 제공
❌ **학습 피드백 루프 없음** - 각 실행이 처음부터 시작

LLMunix는 이 모델을 뒤집습니다:

✅ **무한한 도메인 범위** - 필요한 모든 전문성을 위한 에이전트 생성
✅ **문제 특화 에이전트** - 정확한 요구사항에 맞춤화된 프롬프트
✅ **최소한의 코어** - 3개의 시스템 에이전트만 제공
✅ **지속적 진화** - 모든 프로젝트가 미래 성능을 향상시킴

## 제공되는 것

이 템플릿에는 **필수 커널 구성요소만** 포함되어 있습니다:

### 핵심 시스템 에이전트 (3개만)
- **SystemAgent** - 멀티 에이전트 조율을 위한 마스터 오케스트레이터
- **MemoryAnalysisAgent** - 메모리 로그를 분석하고 해석합니다
- **MemoryConsolidationAgent** - 단기 메모리를 장기 학습으로 통합합니다

### 시스템 명세
- **SmartMemory.md** - 계층적 메모리 아키텍처
- **도구 정의** - LLMunix 개념이 Claude Code 도구에 매핑되는 방법
- **메모리 관리자** - 쿼리 및 추적 관리 시스템

### 커널 (CLAUDE.md)
- Claude Code 시작 시 자동 로드됨
- SystemAgent 오케스트레이션 워크플로우 구현
- 동적 에이전트 생성 및 실행 처리

**그게 전부입니다.** 예제 프로젝트 없음. 도메인 특화 에이전트 없음. 비대함 없음.

Claude에게 목표를 주면, 필요한 에이전트를 정확히 생성합니다—그 이상도 이하도 없이.

## 빠른 시작

### 공개 저장소의 경우

1. **이 템플릿 사용** → "Use this template"을 클릭하고 공개 저장소 생성
2. **Claude Code에 연결** → [claude.ai/code](https://claude.ai/code)를 방문하여 GitHub 인증
3. **저장소 선택** → 목록에서 새 저장소 선택
4. **Claude에게 목표 제공** → 야심 차고 다면적인 문제로 시작
5. **검토 및 병합** → Claude가 변경사항을 PR 검토를 위한 브랜치에 푸시

### 비공개 저장소의 경우

1. **이 템플릿 사용** → 이 템플릿에서 **비공개** 저장소 생성
2. **GitHub 앱 구성** → Claude에게 비공개 저장소 접근 권한 부여
3. **환경 변수 설정** → 시크릿 및 API 키를 안전하게 구성
4. **선택 사항: SessionStart 훅 추가** → 의존성 설치 자동화
5. **프로젝트 시작** → 완전한 격리 상태에서 독점 코드 작업

### 상세 설정 가이드

공개 및 비공개 저장소 모두에 대한 완전한 단계별 지침은 **[GETTING_STARTED.md](GETTING_STARTED.md)**를 참조하세요.

이 가이드에서 다루는 내용:
- 공개 vs 비공개 저장소 설정
- 네트워크 접근 및 보안 구성
- 환경 변수 및 시크릿 관리
- SessionStart 훅으로 의존성 설치
- 웹과 로컬 CLI 간 세션 이동
- 일반적인 문제 해결

## 작동 방식: 동적 진화

### 워크플로우

```
사용자 목표
    ↓
1. SystemAgent가 목표를 분석하고 분해
    ↓
2. projects/[ProjectName]/에 프로젝트 구조 생성
    ↓
3. 필요한 전문성을 위한 새 에이전트 마크다운 파일 작성
    ↓
4. 각 에이전트를 읽고 Task 도구로 호출
    ↓
5. memory/short_term/에 모든 상호작용 기록
    ↓
6. output/에 최종 출력물 생성
    ↓
7. memory/long_term/에 학습 내용 통합
    ↓
8. 향후 프로젝트가 이 학습 내용을 쿼리하고 재사용
```

### 에이전트 생성 예시

머신러닝 파이프라인을 요청하면, LLMunix는 다음을 생성할 수 있습니다:

```markdown
---
name: FeatureEngineerAgent
type: specialist
capabilities:
  - 특성 선택 및 추출
  - 차원 축소
  - 특성 인코딩
tools: [Read, Write, Bash]
---

당신은 다음을 전문으로 하는 전문 특성 엔지니어입니다...
[상세하고 프로젝트 특화된 프롬프트]
```

이 에이전트는:
- **런타임에 생성됨** - 이 특정 프로젝트를 위해
- **맞춤화됨** - 정확한 ML 문제 도메인에
- **기록됨** - 향후 재사용을 위해
- **일시적** - 필요할 때만 존재

## 사용 사례 예시

### 양자 컴퓨팅 연구
```
목표: "공급망 최적화를 위한 양자 어닐링 알고리즘 개발"

동적으로 생성된 에이전트:
- LogisticsVisionaryAgent.md
- QuantumAlgorithmDesignerAgent.md
- QiskitImplementationAgent.md
- OptimizationValidatorAgent.md
```

### 풀스택 개발
```
목표: "WebSocket 동기화가 있는 실시간 협업 화이트보드 구축"

동적으로 생성된 에이전트:
- SystemArchitectAgent.md
- ReactFrontendAgent.md
- WebSocketBackendAgent.md
- DatabaseSchemaAgent.md
- IntegrationTestAgent.md
```

### 데이터 과학
```
목표: "행동 분석을 사용한 고객 이탈 예측"

동적으로 생성된 에이전트:
- DataExplorationAgent.md
- FeatureEngineeringAgent.md
- ModelTrainingAgent.md
- ResultsVisualizationAgent.md
```

## 아키텍처

### 프로젝트 구조 (동적으로 생성됨)

```
llmunix-starter/
├── .claude/
│   └── agents/                    # 3개의 핵심 에이전트만
│       ├── SystemAgent.md
│       ├── MemoryAnalysisAgent.md
│       └── MemoryConsolidationAgent.md
├── system/
│   ├── agents/                    # 소스 정의
│   ├── tools/                     # 도구 명세
│   └── SmartMemory.md
├── projects/                      # 비어 있음 - 동적으로 생성
│   └── [목표별로 동적 생성]
│       ├── components/
│       │   └── agents/            # 프로젝트 특화 에이전트
│       ├── output/                # 최종 산출물
│       └── memory/
│           ├── short_term/        # 상호작용 로그
│           └── long_term/         # 통합된 학습 내용
├── CLAUDE.md                      # 커널 (자동 로드)
└── README.md
```

### 메모리 시스템

**단기 메모리 (에피소드)**
- 프로젝트 런타임 동안의 원시 실행 추적
- 전체 프롬프트와 응답이 포함된 타임스탬프 로그
- 에이전트 상호작용당 하나의 파일

**장기 메모리 (시맨틱)**
- 향후 재사용을 위해 정제된 지식
- 에이전트 템플릿 및 워크플로우 패턴
- 도메인 인사이트 및 모범 사례

**학습 루프**
```
실행 → 기록 → 통합 → 저장 → 쿼리 → 적용 → 실행...
```

각 프로젝트가 다음 프로젝트를 위해 시스템을 더 똑똑하게 만듭니다.

## 왜 순수 마크다운인가?

LLMunix는 전적으로 마크다운을 통해 운영됩니다:

- **에이전트 정의** = YAML 프론트매터가 있는 마크다운
- **메모리 추적** = 마크다운 상호작용 로그
- **지식 베이스** = 마크다운 학습 문서

이점:
- **사람이 읽을 수 있음** - 모든 시스템 상태가 검사 가능
- **버전 제어 가능** - Git에서 진화 추적
- **이식 가능** - 바이너리 의존성이나 데이터베이스 없음
- **LLM 네이티브** - Claude가 자연스럽게 읽고 씁니다

## 모범 사례

### 명확하고 야심 찬 목표 작성

❌ 모호함: "코드 좀 도와주세요"
✅ 명확함: "Redis 백엔드, 지수 백오프, 모니터링 대시보드가 있는 분산 작업 큐 구축"

❌ 너무 좁음: "배열을 정렬하는 함수 작성"
✅ 야심 찬: "스트리밍 수집, 변환, 실시간 분석이 포함된 데이터 처리 파이프라인 생성"

### 시스템을 신뢰하세요

- 어떤 에이전트를 생성할지 마이크로매니지하지 마세요
- LLMunix가 최적의 분해를 결정하도록 하세요
- 나중에 생성된 에이전트를 검토하여 패턴을 학습하세요

### 학습 활용

몇 개의 프로젝트 후:
- `memory/long_term/`을 확인하여 시스템이 무엇을 배웠는지 확인
- 유사한 향후 프로젝트가 더 빠르게 부트스트랩됨
- 에이전트 템플릿이 시간이 지남에 따라 더 정교해짐

## 고급: 순수 마크다운 패러다임

모든 구성요소가 마크다운입니다:

**1. 에이전트 정의 = 실행 가능한 프롬프트**
```markdown
---
name: MyAgent
capabilities: [...]
---
시스템 프롬프트가 여기에...
```

**2. 오케스트레이션 = 읽기 및 위임**
```
1. 에이전트 정의 작성
2. 에이전트 정의 읽기
3. Task 도구에 내용 전달
```

**3. 학습 = 마크다운 분석**
```
MemoryConsolidationAgent가 short_term/*.md 읽기
패턴 추출 → long_term/learnings.md 작성
```

## 비교: 플러그인 vs 스타터

이 스타터 템플릿은 [LLMunix CLI 플러그인](https://github.com/evolving-agents-labs/llmunix-marketplace)의 **저장소 버전**입니다.

| 기능 | CLI 플러그인 | 스타터 템플릿 |
|---------|------------|------------------|
| 설치 | `/plugin install` | "Use this template" |
| 활성화 | `/llmunix "목표"` | Claude와 대화만 하면 됨 |
| 핵심 에이전트 | 3개만 | 3개만 |
| 프로젝트 | 동적 | 동적 |
| 학습 | 예 | 예 |
| 최적 대상 | 로컬 CLI 사용자 | 웹 사용자 |

둘 다 같은 철학을 공유합니다: **최소한의 코어, 무한한 확장성**.

## 문제 해결

### 문제: 에이전트 생성 품질이 다양함
**해결책:** 프로젝트 후, `components/agents/*.md`를 검토하고 프롬프트를 수동으로 개선하세요. 다음 통합이 개선된 템플릿을 저장합니다.

### 문제: 과거 학습을 재사용하지 않음
**해결책:** 목표에 도메인 특화 키워드를 사용하세요. `long_term/`을 확인하여 어떤 지식이 있는지 확인하세요.

### 문제: 너무 많은 에이전트가 생성됨
**해결책:** 이것은 실제로 괜찮습니다—시스템이 철저한 것입니다. 시간이 지나면 유사한 역할을 통합하는 법을 배웁니다.

## 이 템플릿에 포함되지 않은 것

최소주의 철학에 따라:
- ❌ 예제 프로젝트 없음
- ❌ 도메인 특화 에이전트 없음
- ❌ 사전 구축된 솔루션 없음
- ❌ 샘플 출력물이나 메모리 로그 없음

이것들이 필요하면, 직접 만드세요. 그것이 요점입니다.

## 기여

**핵심 커널**을 향상시키는 기여를 환영합니다:

- 핵심 에이전트 프롬프트 개선
- 시스템 명세 향상
- 메모리 관리 최적화
- 버그 보고 및 수정

**제출하지 마세요:** 사전 구축된 도메인 에이전트나 예제 프로젝트. 그것은 목적에 어긋납니다.

## 라이선스

Apache License 2.0 - 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 크레딧

**Evolving Agents Labs** - LLMunix 제작자

Anthropic의 [Claude Code](https://claude.ai/code)를 기반으로 구축되었습니다.

## 링크

- **시작 가이드**: [GETTING_STARTED.md](GETTING_STARTED.md) - 공개 및 비공개 저장소를 위한 완전한 설정
- **문서**: `CLAUDE_CODE_ARCHITECTURE.md`, `system/SmartMemory.md`
- **플러그인 버전**: [LLMunix CLI 플러그인](https://github.com/evolving-agents-labs/llmunix-marketplace)
- **Claude Code 웹**: [공식 문서](https://docs.claude.com/en/docs/claude-code/claude-code-on-the-web)
- **Claude Code**: [공식 웹사이트](https://claude.ai/code)

---

**빌드를 시작하세요:**

Claude에게 야심 차고 복잡한 목표를 주세요. 그것을 해결하기 위한 완벽한 에이전트 팀을 만드는 것을 지켜보세요. 그 에이전트들이 다음 목표를 위한 템플릿이 되는 것을 지켜보세요.

**이것이 실제 동작하는 동적 진화입니다.**
