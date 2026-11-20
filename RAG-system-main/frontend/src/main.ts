
import './style.css'

const app = document.querySelector<HTMLDivElement>('#app')

if (!app) {
  throw new Error('App container #app was not found in index.html')
}

const API_URL = 'http://127.0.0.1:8000'
const SESSION_ID = `finrag-${Date.now()}`
const MAX_FILE_SIZE = 20 * 1024 * 1024

const ALLOWED_EXTENSIONS =
  /\.(pdf|txt|docx|png|jpg|jpeg|bmp|tif|tiff|webp|mp3|wav|m4a|flac|ogg)$/i

let accessToken: string | null = null
let selectedFiles: File[] = []
let conversationCount = 0
let isUploading = false
let isAsking = false

// --------------------------------------------------
// APPLICATION HTML
// --------------------------------------------------

app.innerHTML = `
  <section id="login-screen" class="login-screen">
    <div class="login-card">

      <div class="brand-mark">F</div>

      <h1>FinRAG AI</h1>

      <p class="muted">
        Your financial document intelligence assistant
      </p>

      <form id="login-form">

        <label for="username">Username</label>

        <input
          id="username"
          type="text"
          placeholder="Enter your username"
          autocomplete="username"
          required
        />

        <label for="password">Password</label>

        <input
          id="password"
          type="password"
          placeholder="Enter your password"
          autocomplete="current-password"
          required
        />

        <button class="primary-btn full-width" type="submit">
          Sign in
        </button>

      </form>

      <p id="login-status" class="status-message" aria-live="polite"></p>

      <p class="login-note" id="login-note">
        Don't have an account?
        <button id="show-register" class="auth-switch" type="button">
          Create account
        </button>
      </p>

      <form id="register-form" class="hidden">

        <label for="full-name">Full Name</label>

        <input
          id="full-name"
          type="text"
          placeholder="Enter your full name"
          autocomplete="name"
          minlength="2"
          maxlength="100"
          required
        />

        <label for="register-username">Username</label>

        <input
          id="register-username"
          type="text"
          placeholder="Choose a username"
          autocomplete="username"
          minlength="3"
          maxlength="30"
          required
        />

        <label for="email">Email Address</label>

        <input
          id="email"
          type="email"
          placeholder="Enter your email"
          autocomplete="email"
          required
        />

        <label for="register-password">Password</label>

        <input
          id="register-password"
          type="password"
          placeholder="Create a password"
          autocomplete="new-password"
          minlength="8"
          maxlength="128"
          required
        />

        <label for="confirm-password">Confirm Password</label>

        <input
          id="confirm-password"
          type="password"
          placeholder="Confirm your password"
          autocomplete="new-password"
          minlength="8"
          maxlength="128"
          required
        />

        <button class="primary-btn full-width" type="submit">
          Create account
        </button>

      </form>

      <p id="register-status" class="status-message" aria-live="polite"></p>

      <p id="register-note" class="login-note hidden">
        Already have an account?
        <button id="show-login" class="auth-switch" type="button">
          Sign in
        </button>
      </p>

    </div>
  </section>

  <!-- DASHBOARD -->

  <section id="dashboard" class="dashboard hidden">

    <header class="topbar">

      <div class="brand">

        <div class="brand-mark small">F</div>

        <div>
          <strong>FinRAG AI</strong>
          <span>Financial Document Intelligence</span>
        </div>

      </div>

      <div class="topbar-actions">

        <span class="system-status">
          <span class="status-dot"></span>
          Local AI
        </span>

        <button id="logout-btn" class="secondary-btn" type="button">
          Sign out
        </button>

      </div>

    </header>

    <main class="workspace">

      <div class="page-heading">

        <div>
          <p class="eyebrow">YOUR WORKSPACE</p>

          <h2>Financial Intelligence Dashboard</h2>

          <p class="muted">
            Upload documents and ask questions about their contents.
          </p>
        </div>

      </div>

      <div class="dashboard-grid">

        <!-- DOCUMENT UPLOAD PANEL -->

        <section class="panel documents-panel">

          <div class="panel-heading">

            <div>
              <h3>Documents</h3>
              <p class="muted">
                Upload and index financial files
              </p>
            </div>

            <div class="document-counts">
              <span id="uploaded-file-count" class="count-badge">
                0 uploaded
              </span>

              <span id="file-count" class="count-badge">
                0 selected
              </span>
            </div>

          </div>

          <label id="drop-zone" class="drop-zone" for="file-input">

            <div class="upload-icon">↑</div>

            <strong>Drop files here</strong>

            <span>or browse from your computer</span>

            <small>
              TXT, PDF, DOCX, images and audio — max 20 MB per file
            </small>

          </label>

          <input
            id="file-input"
            type="file"
            multiple
            accept=".pdf,.txt,.docx,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp,.mp3,.wav,.m4a,.flac,.ogg"
            hidden
          />

          <div class="section-label">Selected files</div>

          <div id="file-list" class="file-list">

            <div class="empty-state">
              <span class="empty-icon">□</span>
              <p>No files selected yet</p>
            </div>

          </div>

          <button
            id="upload-btn"
            class="primary-btn full-width"
            type="button"
            disabled
          >
            Upload documents
          </button>

          <p id="upload-status" class="status-message" aria-live="polite">
            Select supported files to get started.
          </p>

          <div class="info-box">

            <strong>How it works</strong>

            <p>1. Select documents, images, or audio recordings.</p>
            <p>2. FinRAG extracts text or transcribes speech locally.</p>
            <p>3. Ask questions about the indexed content.</p>
            <p>4. Existing ChromaDB data is retained.</p>

          </div>

        </section>

        <!-- QUESTION ANSWERING PANEL -->

        <section class="panel chat-panel">

          <div class="panel-heading">

            <div>
              <h3>Ask FinRAG AI</h3>

              <p class="muted">
                Get answers grounded in your documents
              </p>
            </div>

            <span class="ai-label">Offline RAG</span>

          </div>

          <div id="welcome" class="welcome-state">

            <div class="welcome-icon">✧</div>

            <h3>What would you like to know?</h3>

            <p class="muted">
              Ask a question about information in your indexed documents.
            </p>

            <div class="suggestions">

              <button
                class="suggestion"
                type="button"
                data-question="What was the company's revenue in 2025?"
              >
                What was the revenue in 2025?
              </button>

              <button
                class="suggestion"
                type="button"
                data-question="How much did net profit increase from 2024 to 2025?"
              >
                Compare annual net profit
              </button>

              <button
                class="suggestion"
                type="button"
                data-question="Summarize the financial report."
              >
                Summarize the financial report
              </button>

            </div>

          </div>

          <div
            id="conversation"
            class="conversation hidden"
            aria-live="polite"
          ></div>

          <form id="question-form" class="question-form">

            <textarea
              id="question-input"
              placeholder="Ask a question about your documents..."
              rows="2"
              required
            ></textarea>

            <div class="composer-footer">

              <span>
                Answers are based on retrieved document content.
              </span>

              <button id="ask-btn" class="primary-btn" type="submit">
                Ask AI →
              </button>

            </div>

          </form>

          <p id="ask-status" class="status-message" aria-live="polite"></p>

        </section>

      </div>

    </main>

  </section>
`

// --------------------------------------------------
// DOM ELEMENTS
// --------------------------------------------------

const loginScreen =
  document.querySelector<HTMLElement>('#login-screen')!

const dashboard =
  document.querySelector<HTMLElement>('#dashboard')!

const loginForm =
  document.querySelector<HTMLFormElement>('#login-form')!

const usernameInput =
  document.querySelector<HTMLInputElement>('#username')!

const passwordInput =
  document.querySelector<HTMLInputElement>('#password')!

const loginStatus =
  document.querySelector<HTMLElement>('#login-status')!

const loginNote =
  document.querySelector<HTMLElement>('#login-note')!

const registerForm =
  document.querySelector<HTMLFormElement>('#register-form')!

const registerStatus =
  document.querySelector<HTMLElement>('#register-status')!

const registerNote =
  document.querySelector<HTMLElement>('#register-note')!

const showRegisterButton =
  document.querySelector<HTMLButtonElement>('#show-register')!

const showLoginButton =
  document.querySelector<HTMLButtonElement>('#show-login')!

const fullNameInput =
  document.querySelector<HTMLInputElement>('#full-name')!

const registerUsernameInput =
  document.querySelector<HTMLInputElement>('#register-username')!

const emailInput =
  document.querySelector<HTMLInputElement>('#email')!

const registerPasswordInput =
  document.querySelector<HTMLInputElement>('#register-password')!

const confirmPasswordInput =
  document.querySelector<HTMLInputElement>('#confirm-password')!

const logoutButton =
  document.querySelector<HTMLButtonElement>('#logout-btn')!

const fileInput =
  document.querySelector<HTMLInputElement>('#file-input')!

const dropZone =
  document.querySelector<HTMLElement>('#drop-zone')!

const fileList =
  document.querySelector<HTMLElement>('#file-list')!

const fileCount =
  document.querySelector<HTMLElement>('#file-count')!

const uploadedFileCount =
  document.querySelector<HTMLElement>('#uploaded-file-count')!

const uploadButton =
  document.querySelector<HTMLButtonElement>('#upload-btn')!

const uploadStatus =
  document.querySelector<HTMLElement>('#upload-status')!

const questionForm =
  document.querySelector<HTMLFormElement>('#question-form')!

const questionInput =
  document.querySelector<HTMLTextAreaElement>('#question-input')!

const askButton =
  document.querySelector<HTMLButtonElement>('#ask-btn')!

const askStatus =
  document.querySelector<HTMLElement>('#ask-status')!

const welcome =
  document.querySelector<HTMLElement>('#welcome')!

const conversation =
  document.querySelector<HTMLElement>('#conversation')!

// --------------------------------------------------
// RESPONSE HELPERS
// --------------------------------------------------

async function readResponse(
  response: Response
): Promise<Record<string, unknown>> {

  const text = await response.text()

  if (!text) return {}

  try {

    const result: unknown = JSON.parse(text)

    if (typeof result === 'object' && result !== null) {
      return result as Record<string, unknown>
    }

    return {}

  } catch {

    return { detail: text }

  }
}

function getErrorMessage(
  result: Record<string, unknown>,
  fallback: string
): string {

  if (typeof result.detail === 'string') {
    return result.detail
  }

  if (typeof result.message === 'string') {
    return result.message
  }

  return fallback
}

function authHeaders(): HeadersInit {

  if (!accessToken) {
    throw new Error('Your session has expired. Please sign in again.')
  }

  return {
    Authorization: `Bearer ${accessToken}`,
  }
}

async function refreshUploadedFileCount(): Promise<void> {

  const response = await fetch(`${API_URL}/uploads/count`, {
    method: 'GET',
    headers: authHeaders(),
  })

  const result = await readResponse(response)

  if (!response.ok) {
    throw new Error(
      getErrorMessage(
        result,
        `Could not load uploaded file count (HTTP ${response.status}).`
      )
    )
  }

  if (typeof result.uploaded_files !== 'number') {
    throw new Error(
      'The backend did not return the uploaded file count.'
    )
  }

  uploadedFileCount.textContent =
    `${result.uploaded_files} uploaded`

}

// --------------------------------------------------
// AUTHENTICATION SCREEN MANAGEMENT
// --------------------------------------------------

function showLogin(message = '') {

  dashboard.classList.add('hidden')
  loginScreen.classList.remove('hidden')

  loginForm.classList.remove('hidden')
  registerForm.classList.add('hidden')

  loginNote.classList.remove('hidden')
  registerNote.classList.add('hidden')

  loginStatus.textContent = message

  loginStatus.className = message
    ? 'status-message warning'
    : 'status-message'

}

function showRegistration() {

  loginForm.classList.add('hidden')
  registerForm.classList.remove('hidden')

  loginNote.classList.add('hidden')
  registerNote.classList.remove('hidden')

  loginStatus.textContent = ''
  registerStatus.textContent = ''

  registerUsernameInput.focus()

}

function showLoginForm(message = '') {

  registerForm.classList.add('hidden')
  loginForm.classList.remove('hidden')

  registerNote.classList.add('hidden')
  loginNote.classList.remove('hidden')

  registerStatus.textContent = ''

  loginStatus.textContent = message

  loginStatus.className = message
    ? 'status-message success'
    : 'status-message'

}

function handleUnauthorized() {

  accessToken = null

  resetDashboard()

  showLogin('Your session has expired. Please sign in again.')

}

// --------------------------------------------------
// RESET DASHBOARD
// --------------------------------------------------

function resetDashboard() {

  selectedFiles = []
  conversationCount = 0
  isUploading = false
  isAsking = false

  renderFiles()
  uploadedFileCount.textContent = '0 uploaded'

  conversation.replaceChildren()

  conversation.classList.add('hidden')
  welcome.classList.remove('hidden')

  uploadStatus.textContent =
    'Select supported files to get started.'

  uploadStatus.className = 'status-message'

  askStatus.textContent = ''
  askStatus.className = 'status-message'

  questionInput.value = ''

  askButton.disabled = false
  askButton.textContent = 'Ask AI →'

  uploadButton.textContent = 'Upload documents'

}

// --------------------------------------------------
// REGISTRATION
// --------------------------------------------------

showRegisterButton.addEventListener('click', showRegistration)

showLoginButton.addEventListener('click', () => {
  showLoginForm()
})

registerForm.addEventListener('submit', async (event) => {

  event.preventDefault()

  const fullName = fullNameInput.value.trim()
  const username = registerUsernameInput.value.trim()
  const email = emailInput.value.trim()

  const password = registerPasswordInput.value
  const confirmPassword = confirmPasswordInput.value

  if (password !== confirmPassword) {

    registerStatus.textContent = 'Passwords do not match.'
    registerStatus.className = 'status-message warning'

    return
  }

  if (password.length < 8 || password.length > 128) {

    registerStatus.textContent =
      'Password must contain between 8 and 128 characters.'

    registerStatus.className = 'status-message warning'

    return
  }

  const submitButton =
    registerForm.querySelector<HTMLButtonElement>(
      'button[type="submit"]'
    )

  if (!submitButton) return

  submitButton.disabled = true
  submitButton.textContent = 'Creating account...'

  registerStatus.textContent = ''
  registerStatus.className = 'status-message'

  try {

    const response = await fetch(`${API_URL}/register`, {

      method: 'POST',

      headers: {
        'Content-Type': 'application/json',
      },

      body: JSON.stringify({
        full_name: fullName,
        username,
        email,
        password,
        confirm_password: confirmPassword,
      }),

    })

    const result = await readResponse(response)

    if (!response.ok) {

      throw new Error(
        getErrorMessage(
          result,
          `Registration failed (HTTP ${response.status}).`
        )
      )
    }

    registerForm.reset()
    usernameInput.value = username

    showLoginForm(
      'Account created successfully! Please sign in.'
    )

  } catch (error) {

    registerStatus.textContent =
      error instanceof Error
        ? error.message
        : 'Unable to register. Please try again.'

    registerStatus.className = 'status-message warning'

  } finally {

    submitButton.disabled = false
    submitButton.textContent = 'Create account'

  }

})

// --------------------------------------------------
// LOGIN
// --------------------------------------------------

loginForm.addEventListener('submit', async (event) => {

  event.preventDefault()

  const username = usernameInput.value.trim()
  const password = passwordInput.value

  if (!username || !password) {

    loginStatus.textContent =
      'Enter your username and password.'

    loginStatus.className = 'status-message warning'

    return
  }

  const submitButton =
    loginForm.querySelector<HTMLButtonElement>(
      'button[type="submit"]'
    )

  if (!submitButton) return

  submitButton.disabled = true
  submitButton.textContent = 'Signing in...'

  loginStatus.textContent = ''
  loginStatus.className = 'status-message'

  try {

    const response = await fetch(`${API_URL}/login`, {

      method: 'POST',

      headers: {
        'Content-Type': 'application/json',
      },

      body: JSON.stringify({
        username,
        password,
      }),

    })

    const result = await readResponse(response)

    if (!response.ok) {

      throw new Error(
        getErrorMessage(
          result,
          `Login failed (HTTP ${response.status}).`
        )
      )
    }

    if (typeof result.access_token !== 'string') {

      throw new Error(
        'The backend did not return an access token.'
      )
    }

    accessToken = result.access_token

    passwordInput.value = ''

    resetDashboard()

    loginScreen.classList.add('hidden')
    dashboard.classList.remove('hidden')

    void refreshUploadedFileCount().catch((error: unknown) => {
      uploadedFileCount.textContent = 'Count unavailable'
      uploadStatus.textContent =
        error instanceof Error
          ? error.message
          : 'Could not load uploaded file count.'
      uploadStatus.className = 'status-message warning'
    })

  } catch (error) {

    loginStatus.textContent =
      error instanceof Error
        ? error.message
        : 'Unable to connect to the backend.'

    loginStatus.className = 'status-message warning'

  } finally {

    submitButton.disabled = false
    submitButton.textContent = 'Sign in'

  }

})

// --------------------------------------------------
// LOGOUT
// --------------------------------------------------

logoutButton.addEventListener('click', () => {

  accessToken = null

  resetDashboard()

  usernameInput.value = ''
  passwordInput.value = ''

  registerForm.reset()

  showLogin()

})

// --------------------------------------------------
// FILE HELPERS
// --------------------------------------------------

function formatFileSize(bytes: number): string {

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`

}

function getFileExtension(filename: string): string {

  const parts = filename.toLowerCase().split('.')

  return parts.length > 1
    ? `.${parts.pop()}`
    : ''

}

// --------------------------------------------------
// ADD FILES
// --------------------------------------------------

function addFiles(files: FileList | File[]) {

  if (isUploading) return

  const rejectedNames: string[] = []

  for (const file of Array.from(files)) {

    if (!ALLOWED_EXTENSIONS.test(file.name)) {

      rejectedNames.push(
        `${file.name} (unsupported format)`
      )

      continue
    }

    if (file.size > MAX_FILE_SIZE) {

      rejectedNames.push(
        `${file.name} (over 20 MB)`
      )

      continue
    }

    if (file.size === 0) {

      rejectedNames.push(
        `${file.name} (empty file)`
      )

      continue
    }

    const exists = selectedFiles.some(
      (item) =>
        item.name === file.name &&
        item.size === file.size &&
        item.lastModified === file.lastModified
    )

    if (!exists) {
      selectedFiles.push(file)
    }

  }

  renderFiles()

  if (rejectedNames.length > 0) {

    uploadStatus.textContent =
      `Not added: ${rejectedNames.join(', ')}. Choose supported, non-empty files under 20 MB.`

    uploadStatus.className = 'status-message warning'

  } else {

    uploadStatus.textContent =
      selectedFiles.length > 0
        ? 'Files selected. Click Upload documents to index them.'
        : 'Select supported files to get started.'

    uploadStatus.className = 'status-message'

  }

}

// --------------------------------------------------
// RENDER FILES
// --------------------------------------------------

function renderFiles() {

  fileCount.textContent =
    `${selectedFiles.length} selected`

  uploadButton.disabled =
    selectedFiles.length === 0 || isUploading

  if (selectedFiles.length === 0) {

    fileList.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">□</span>
        <p>No files selected yet</p>
      </div>
    `

    return
  }

  fileList.replaceChildren()

  selectedFiles.forEach((file, index) => {

    const item = document.createElement('div')
    item.className = 'file-item'

    const details = document.createElement('div')
    details.className = 'file-details'

    const name = document.createElement('strong')
    name.textContent = file.name

    const size = document.createElement('span')

    size.textContent =
      `${formatFileSize(file.size)} · ${getFileExtension(file.name).slice(1).toUpperCase()}`

    details.append(name, size)

    const remove = document.createElement('button')

    remove.className = 'remove-file'
    remove.type = 'button'
    remove.textContent = '×'
    remove.disabled = isUploading

    remove.setAttribute(
      'aria-label',
      `Remove ${file.name}`
    )

    remove.addEventListener('click', () => {

      if (isUploading) return

      selectedFiles.splice(index, 1)

      renderFiles()

      uploadStatus.textContent =
        'File removed from the selection.'

      uploadStatus.className = 'status-message'

    })

    item.append(details, remove)
    fileList.append(item)

  })

}

// --------------------------------------------------
// FILE SELECTION
// --------------------------------------------------

fileInput.addEventListener('change', () => {

  if (fileInput.files) {
    addFiles(fileInput.files)
  }

  fileInput.value = ''

})

// --------------------------------------------------
// DRAG AND DROP
// --------------------------------------------------

dropZone.addEventListener('dragover', (event) => {

  event.preventDefault()
  dropZone.classList.add('dragging')

})

dropZone.addEventListener('dragleave', () => {

  dropZone.classList.remove('dragging')

})

dropZone.addEventListener('drop', (event) => {

  event.preventDefault()
  dropZone.classList.remove('dragging')

  if (event.dataTransfer?.files) {
    addFiles(event.dataTransfer.files)
  }

})

// --------------------------------------------------
// UPLOAD DOCUMENTS
// --------------------------------------------------

uploadButton.addEventListener('click', async () => {

  if (isUploading || selectedFiles.length === 0) return

  if (!accessToken) {
    handleUnauthorized()
    return
  }

  isUploading = true

  uploadButton.textContent = 'Uploading...'

  uploadStatus.textContent =
    'Uploading and indexing your documents. Audio may take longer to process.'

  uploadStatus.className = 'status-message'

  renderFiles()

  const filesToUpload = [...selectedFiles]
  const successfulFiles = new Set<File>()

  const failures: string[] = []
  const successes: string[] = []

  try {

    for (const file of filesToUpload) {

      const formData = new FormData()
      formData.append('file', file, file.name)

      try {

        const response = await fetch(`${API_URL}/upload`, {

          method: 'POST',
          headers: authHeaders(),
          body: formData,

        })

        const result = await readResponse(response)

        if (response.status === 401) {

          handleUnauthorized()
          failures.push(`${file.name}: session expired`)

          break
        }

        if (!response.ok) {

          throw new Error(
            getErrorMessage(
              result,
              `Upload failed with HTTP ${response.status}`
            )
          )
        }

        successfulFiles.add(file)

        const filename =
          typeof result.filename === 'string'
            ? result.filename
            : file.name

        const modality =
          typeof result.modality === 'string'
            ? result.modality
            : 'file'

        const chunks =
          typeof result.chunks_added === 'number'
            ? result.chunks_added
            : 0

        successes.push(
          `${filename} (${modality}): ${chunks} chunks indexed`
        )

      } catch (error) {

        const reason =
          error instanceof Error
            ? error.message
            : 'Unknown error'

        failures.push(`${file.name}: ${reason}`)

      }

    }

  } finally {

    selectedFiles = selectedFiles.filter(
      (file) => !successfulFiles.has(file)
    )

    isUploading = false

    uploadButton.textContent = 'Upload documents'

    renderFiles()

  }

  if (successes.length > 0 && failures.length === 0) {

    uploadStatus.textContent =
      `Upload successful! ${successes.length} file(s) uploaded and indexed. You can now ask questions.`

    uploadStatus.className = 'status-message success'

  } else if (successes.length > 0 && failures.length > 0) {

    uploadStatus.textContent =
      `Some files uploaded successfully: ${successes.length} file(s). Failed: ${failures.length} file(s). Details: ${failures.join(' | ')}`

    uploadStatus.className = 'status-message warning'

  } else if (failures.length > 0) {

    uploadStatus.textContent =
      `Upload failed: ${failures.join(' | ')}`

    uploadStatus.className = 'status-message warning'

  } else {

    uploadStatus.textContent = 'No files were uploaded.'
    uploadStatus.className = 'status-message warning'

  }

  try {
    await refreshUploadedFileCount()
  } catch (error) {
    const reason =
      error instanceof Error
        ? error.message
        : 'Could not load uploaded file count.'

    uploadStatus.textContent =
      `${uploadStatus.textContent} ${reason}`
    uploadStatus.className = 'status-message warning'
  }

})

// --------------------------------------------------
// DISPLAY CHAT MESSAGES
// --------------------------------------------------

function addMessage(
  role: 'user' | 'assistant',
  text: string,
  documentsRetrieved?: number
) {

  const message = document.createElement('article')
  message.className = `message ${role}`

  const label = document.createElement('div')
  label.className = 'message-label'

  label.textContent =
    role === 'user'
      ? 'You'
      : 'FinRAG AI'

  const content = document.createElement('div')
  content.className = 'message-content'
  content.textContent = text

  message.append(label, content)

  if (
    role === 'assistant' &&
    documentsRetrieved !== undefined
  ) {

    const meta = document.createElement('div')
    meta.className = 'answer-meta'

    meta.textContent =
      `Documents retrieved: ${documentsRetrieved}`

    message.append(meta)

  }

  conversation.append(message)

  conversation.scrollTop =
    conversation.scrollHeight

}

// --------------------------------------------------
// ASK QUESTION
// --------------------------------------------------

async function askQuestion(question: string) {

  const cleanedQuestion = question.trim()

  if (!cleanedQuestion || isAsking) return

  if (!accessToken) {
    handleUnauthorized()
    return
  }

  isAsking = true

  welcome.classList.add('hidden')
  conversation.classList.remove('hidden')

  addMessage('user', cleanedQuestion)

  questionInput.value = ''

  askButton.disabled = true
  askButton.textContent = 'Thinking...'

  askStatus.textContent =
    'Searching your indexed documents...'

  askStatus.className = 'status-message'

  const waiting = document.createElement('div')

  waiting.className = 'thinking-message'
  waiting.textContent =
    'FinRAG AI is preparing your answer...'

  conversation.append(waiting)

  conversation.scrollTop =
    conversation.scrollHeight

  try {

    const response = await fetch(`${API_URL}/ask`, {

      method: 'POST',

      headers: {
        ...authHeaders(),
        'Content-Type': 'application/json',
      },

      body: JSON.stringify({
        question: cleanedQuestion,
        session_id: SESSION_ID,
      }),

    })

    const result = await readResponse(response)

    if (response.status === 401) {

      waiting.remove()
      handleUnauthorized()

      return
    }

    if (!response.ok) {

      throw new Error(
        getErrorMessage(
          result,
          `Backend returned HTTP ${response.status}`
        )
      )
    }

    waiting.remove()

    const answer =
      typeof result.answer === 'string'
        ? result.answer
        : 'Information not found.'

    const documentsRetrieved =
      typeof result.documents_retrieved === 'number'
        ? result.documents_retrieved
        : 0

    addMessage(
      'assistant',
      answer,
      documentsRetrieved
    )

    conversationCount++

    askStatus.textContent =
      `Question ${conversationCount} answered.`

    askStatus.className = 'status-message success'

  } catch (error) {

    waiting.remove()

    const reason =
      error instanceof Error
        ? error.message
        : 'Unknown error'

    addMessage(
      'assistant',
      `Unable to get an answer. Check that the FastAPI backend is running at ${API_URL}. Details: ${reason}`
    )

    askStatus.textContent =
      'Could not get an answer.'

    askStatus.className = 'status-message warning'

  } finally {

    isAsking = false

    askButton.disabled = false
    askButton.textContent = 'Ask AI →'

    questionInput.focus()

  }

}

// --------------------------------------------------
// QUESTION FORM
// --------------------------------------------------

questionForm.addEventListener('submit', (event) => {

  event.preventDefault()

  void askQuestion(questionInput.value)

})

// --------------------------------------------------
// SUGGESTED QUESTIONS
// --------------------------------------------------

document
  .querySelectorAll<HTMLButtonElement>('.suggestion')
  .forEach((button) => {

    button.addEventListener('click', () => {

      const question =
        button.dataset.question ?? ''

      if (!isAsking) {

        questionInput.value = question

        void askQuestion(question)

      }

    })

  })
