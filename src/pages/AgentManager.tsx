import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  deleteDocument,
  getAgentPrompts,
  getDocuments,
  getVectorStoreInfo,
  ingestKnowledge,
  rebuildVectorstore,
  saveAgentPrompt,
  AgentPrompt,
  DocumentListItem,
  VectorStoreInfo,
} from "@/utils/api";
import { toast } from "sonner";
import { Database, FileText, Settings2, Trash2 } from "lucide-react";

const AgentManager = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("lcb_user") || "{}") as { role?: string };
  const [prompts, setPrompts] = useState<AgentPrompt[]>([]);
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [vectorInfo, setVectorInfo] = useState<VectorStoreInfo | null>(null);
  const [chunkSearch, setChunkSearch] = useState("");
  const [chunkPage, setChunkPage] = useState(1);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [selectedPromptKey, setSelectedPromptKey] = useState<string>("");
  const [promptDisplayName, setPromptDisplayName] = useState("");
  const [promptText, setPromptText] = useState("");
  const [loading, setLoading] = useState(false);
  const [savingPrompt, setSavingPrompt] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [websiteUrls, setWebsiteUrls] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (user.role !== "admin") return;
    loadData();
  }, [user.role]);

  const selectedPrompt = useMemo(
    () => prompts.find((prompt) => prompt.agent_key === selectedPromptKey),
    [prompts, selectedPromptKey]
  );
  const promptPlaceholders: Record<string, string> = {
    supervisor: "Required placeholders: {available_routes}, {user_request}",
    llm_judge: "Required placeholders: {agent_name}, {user_question}, {reference_context}, {chatbot_response}",
    competitive_intelligence: "Required placeholders: {relevant_context}, {message}",
  };

  useEffect(() => {
    if (selectedPrompt) {
      setPromptDisplayName(selectedPrompt.display_name);
      setPromptText(selectedPrompt.prompt_text);
    }
  }, [selectedPrompt]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [promptList, documentList, vectorData] = await Promise.all([getAgentPrompts(), getDocuments(), getVectorStoreInfo(1)]);
      setPrompts(promptList);
      setDocuments(documentList);
      setVectorInfo(vectorData);
      setChunkPage(1);
      setSelectedPromptKey((current) => current || promptList[0]?.agent_key || "");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to load agent manager data.");
    } finally {
      setLoading(false);
    }
  };

  const loadChunks = async (page = 1, search = chunkSearch) => {
    setLoadingChunks(true);
    try {
      const data = await getVectorStoreInfo(page, search);
      setVectorInfo(data);
      setChunkPage(page);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to inspect vector chunks.");
    } finally {
      setLoadingChunks(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("lcb_auth_token");
    localStorage.removeItem("lcb_user");
    navigate("/login", { replace: true });
  };

  const handleSavePrompt = async () => {
    if (!selectedPromptKey) {
      toast.error("Select an agent prompt first.");
      return;
    }
    if (!promptDisplayName.trim() || !promptText.trim()) {
      toast.error("Prompt name and text cannot be empty.");
      return;
    }
    setSavingPrompt(true);
    try {
      const updated = await saveAgentPrompt(selectedPromptKey, promptDisplayName.trim(), promptText.trim());
      setPrompts((current) => current.map((item) => (item.agent_key === updated.agent_key ? updated : item)));
      toast.success("Agent prompt saved.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to save agent prompt.");
    } finally {
      setSavingPrompt(false);
    }
  };

  const handleUploadFiles = async () => {
    const files = fileInputRef.current?.files;
    if (!files || files.length === 0) {
      toast.error("Choose at least one file to upload.");
      return;
    }
    setUploading(true);
    try {
      await ingestKnowledge([], Array.from(files));
      fileInputRef.current!.value = "";
      toast.success("Files uploaded successfully. Rebuilding the vector store now.");
      await loadData();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to upload files.");
    } finally {
      setUploading(false);
    }
  };

  const handleAddWebsiteSources = async () => {
    const urls = websiteUrls
      .split(/[,\n]/)
      .map((url) => url.trim())
      .filter(Boolean);
    if (!urls.length) {
      toast.error("Enter at least one website URL.");
      return;
    }
    if (urls.some((url) => !/^https?:\/\//i.test(url))) {
      toast.error("Each URL must start with http:// or https://");
      return;
    }
    setUploading(true);
    try {
      await ingestKnowledge(urls, []);
      setWebsiteUrls("");
      toast.success("Website sources added successfully.");
      await loadData();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to add website sources.");
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (docId: number) => {
    try {
      await deleteDocument(docId);
      toast.success("Document deleted.");
      setDocuments((current) => current.filter((doc) => doc.id !== docId));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to delete document.");
    }
  };

  const handleRebuildVectorstore = async () => {
    setRebuilding(true);
    try {
      await rebuildVectorstore();
      toast.success("Vector database rebuild triggered.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to rebuild vector database.");
    } finally {
      setRebuilding(false);
    }
  };

  if (user.role !== "admin") {
    return <Navigate to="/marketing" replace />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-950 via-slate-950 to-slate-900 text-white overflow-hidden">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-36 -right-20 w-96 h-96 rounded-full bg-emerald-500 opacity-10 blur-3xl animate-blob" style={{ animationDuration: "10s" }} />
        <div className="absolute -bottom-40 -left-24 w-96 h-96 rounded-full bg-cyan-500 opacity-10 blur-3xl animate-blob animation-delay-2000" style={{ animationDuration: "12s" }} />
      </div>

      <div className="relative z-10">
        <header className="max-w-7xl mx-auto px-6 pt-6">
          <nav className="flex items-center justify-between gap-6 rounded-full border border-white/10 bg-white/5 px-4 py-3 shadow-lg shadow-black/10 backdrop-blur-xl">
            <div className="flex items-center gap-3">
              <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-emerald-400/15 text-emerald-200 ring-1 ring-white/10">🌿</div>
              <span className="text-sm font-semibold tracking-wide text-emerald-100">LCB AI Assistant</span>
            </div>
            <div className="flex items-center gap-4 text-sm font-medium text-slate-100">
              <Link to="/" className={location.pathname === "/" ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4" : "text-slate-200 hover:text-white"}>General Chat</Link>
              <Link to="/marketing" className={location.pathname === "/marketing" ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4" : "text-slate-200 hover:text-white"}>Marketing</Link>
              <Link to="/competitors" className={location.pathname === "/competitors" ? "text-white underline decoration-indigo-300 decoration-2 underline-offset-4" : "text-slate-200 hover:text-white"}>Competitors</Link>
              <Link to="/agent-manager" className={location.pathname === "/agent-manager" ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4" : "text-slate-200 hover:text-white"}>Agent Manager</Link>
              <Link to="/tracker" className={location.pathname === "/tracker" ? "text-white underline decoration-emerald-300 decoration-2 underline-offset-4" : "text-slate-200 hover:text-white"}>Admin tracker</Link>
              <Button variant="outline" size="sm" className="border-white/20 bg-white/10 text-white hover:bg-white/20" onClick={logout}>Logout</Button>
            </div>
          </nav>
        </header>

        <main className="max-w-7xl mx-auto px-6 py-8">
          <div className="rounded-[2rem] border border-white/10 bg-slate-950/80 p-8 shadow-[0_40px_120px_-48px_rgba(16,185,129,0.65)] backdrop-blur-xl">
            <div className="flex flex-col gap-8 lg:flex-row lg:items-start">
              <div className="space-y-6 lg:w-1/3">
                <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                  <div className="flex items-center gap-3 text-emerald-300">
                    <Settings2 />
                    <h1 className="text-xl font-semibold text-white">Agent Manager</h1>
                  </div>
                  <p className="mt-4 text-sm leading-6 text-slate-300">
                    Edit and manage prompt templates for your agents, and upload or remove content from the vector database.
                  </p>
                </div>
                <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                  <div className="flex items-center gap-3 text-emerald-300">
                    <Database />
                    <div>
                      <h2 className="text-lg font-semibold text-white">Vector database</h2>
                      <p className="mt-1 text-sm text-slate-400">Upload files or add URLs and rebuild the vector store from here.</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:w-2/3">
                <Tabs defaultValue="prompts">
                  <TabsList>
                    <TabsTrigger value="prompts">Agent prompts</TabsTrigger>
                    <TabsTrigger value="vector">Vector database</TabsTrigger>
                  </TabsList>

                  <TabsContent value="prompts">
                    <div className="space-y-6">
                      <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                        <h2 className="text-lg font-semibold text-white">Saved agent prompts</h2>
                        <div className="mt-4 overflow-auto rounded-2xl border border-white/10 bg-slate-950/80">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Name</TableHead>
                                <TableHead>Key</TableHead>
                                <TableHead>Updated</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {prompts.map((prompt) => (
                                <TableRow key={prompt.id} onClick={() => setSelectedPromptKey(prompt.agent_key)} className={prompt.agent_key === selectedPromptKey ? "bg-emerald-500/10" : ""}>
                                  <TableCell>{prompt.display_name}</TableCell>
                                  <TableCell>{prompt.agent_key}</TableCell>
                                  <TableCell>{new Date(prompt.updated_at).toLocaleString()}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </div>

                      <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                        <h2 className="text-lg font-semibold text-white">Edit prompt</h2>
                        {!selectedPrompt ? (
                          <p className="text-sm text-slate-400">Select a prompt from the list to edit it.</p>
                        ) : (
                          <div className="space-y-4">
                            <div>
                              <label className="mb-2 block text-sm font-medium text-slate-200">Prompt name</label>
                              <Input value={promptDisplayName} onChange={(event) => setPromptDisplayName(event.target.value)} placeholder="Display name" className="bg-slate-950/70 text-white" />
                            </div>
                            <div>
                              <label className="mb-2 block text-sm font-medium text-slate-200">Prompt text</label>
                              <Textarea value={promptText} onChange={(event) => setPromptText(event.target.value)} className="bg-slate-950/70 text-white" rows={14} />
                              {promptPlaceholders[selectedPromptKey] && <p className="mt-2 text-xs text-amber-200">
                                {promptPlaceholders[selectedPromptKey]}
                              </p>}
                            </div>
                            <Button onClick={handleSavePrompt} disabled={savingPrompt} className="rounded-full bg-emerald-600 px-5 py-3 text-sm font-semibold text-white hover:bg-emerald-500">
                              {savingPrompt ? "Saving..." : "Save prompt"}
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="vector">
                    <div className="space-y-6">
                      {vectorInfo && <div className="rounded-[1.75rem] border border-emerald-400/20 bg-emerald-400/5 p-6">
                        <div className="flex flex-wrap items-start justify-between gap-4">
                          <div>
                            <h2 className="text-lg font-semibold text-white">Live vector index inspector</h2>
                            <p className="mt-1 text-sm text-slate-400">Collection: {vectorInfo.status.collection_name} · Mode: {vectorInfo.status.retrieval_mode}</p>
                          </div>
                          <span className={`rounded-full px-3 py-1 text-xs font-semibold ${vectorInfo.status.vectorstore_ready ? "bg-emerald-400/15 text-emerald-200" : "bg-amber-400/15 text-amber-200"}`}>
                            {vectorInfo.status.vectorstore_ready ? "Vector index ready" : "Keyword fallback active"}
                          </span>
                        </div>
                        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                          <div className="rounded-2xl bg-slate-950/70 p-4"><p className="text-xs text-slate-400">Total chunks</p><p className="mt-1 text-2xl font-semibold">{vectorInfo.statistics.total_chunks}</p></div>
                          <div className="rounded-2xl bg-slate-950/70 p-4"><p className="text-xs text-slate-400">Persisted vectors</p><p className="mt-1 text-2xl font-semibold">{vectorInfo.status.persisted_vector_count ?? "—"}</p></div>
                          <div className="rounded-2xl bg-slate-950/70 p-4"><p className="text-xs text-slate-400">Total characters</p><p className="mt-1 text-2xl font-semibold">{vectorInfo.statistics.total_characters.toLocaleString()}</p></div>
                          <div className="rounded-2xl bg-slate-950/70 p-4"><p className="text-xs text-slate-400">Average chunk</p><p className="mt-1 text-2xl font-semibold">{vectorInfo.statistics.average_chunk_characters} chars</p></div>
                        </div>
                        <div className="mt-4 rounded-2xl bg-slate-950/70 p-4 text-xs text-slate-300">
                          <p><span className="font-semibold text-slate-100">Embedding model:</span> {vectorInfo.status.embedding_model}</p>
                          <p className="mt-1 break-all"><span className="font-semibold text-slate-100">Storage:</span> {vectorInfo.status.storage_path}</p>
                          <div className="mt-3 flex flex-wrap gap-2">{Object.entries(vectorInfo.statistics.source_counts).map(([type, count]) => <span key={type} className="rounded-full border border-white/10 px-3 py-1">{type}: {count}</span>)}</div>
                        </div>
                      </div>}

                      <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                        <div className="flex items-center gap-3">
                          <FileText className="text-emerald-300" />
                          <div>
                            <h2 className="text-lg font-semibold text-white">Add knowledge</h2>
                            <p className="mt-1 text-sm text-slate-400">Upload files or add public website pages to the RAG database.</p>
                          </div>
                        </div>

                        <div className="mt-5 space-y-4">
                          <div className="space-y-2">
                            <label className="block text-sm font-medium text-slate-200">Upload documents</label>
                            <input ref={fileInputRef} type="file" multiple className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-white" />
                            <Button onClick={handleUploadFiles} disabled={uploading} className="rounded-full bg-emerald-600 px-5 py-3 text-sm font-semibold text-white hover:bg-emerald-500">
                              {uploading ? "Uploading..." : "Upload files"}
                            </Button>
                          </div>

                          <div className="space-y-2">
                            <label className="block text-sm font-medium text-slate-200">Add website sources</label>
                            <Textarea value={websiteUrls} onChange={(event) => setWebsiteUrls(event.target.value)} placeholder="https://example.com/page" rows={4} className="bg-slate-950/70 text-white" />
                            <Button onClick={handleAddWebsiteSources} disabled={uploading} className="rounded-full bg-emerald-600 px-5 py-3 text-sm font-semibold text-white hover:bg-emerald-500">
                              {uploading ? "Importing..." : "Add website sources"}
                            </Button>
                          </div>

                          <div>
                            <Button onClick={handleRebuildVectorstore} disabled={rebuilding} className="rounded-full bg-slate-800 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-700">
                              {rebuilding ? "Rebuilding..." : "Rebuild vector database"}
                            </Button>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                          <div><h2 className="text-lg font-semibold text-white">All knowledge chunks</h2><p className="mt-1 text-sm text-slate-400">Inspect exact text and metadata used by retrieval.</p></div>
                          <div className="flex gap-2">
                            <Input value={chunkSearch} onChange={(event) => setChunkSearch(event.target.value)} onKeyDown={(event) => event.key === "Enter" && loadChunks(1)} placeholder="Search chunks or sources" className="min-w-64 bg-slate-950/70 text-white" />
                            <Button onClick={() => loadChunks(1)} disabled={loadingChunks} variant="outline">Search</Button>
                          </div>
                        </div>
                        <div className="mt-4 overflow-auto rounded-2xl border border-white/10 bg-slate-950/80">
                          <Table>
                            <TableHeader><TableRow><TableHead>ID / Index</TableHead><TableHead>Source</TableHead><TableHead>Size</TableHead><TableHead>Chunk content and metadata</TableHead></TableRow></TableHeader>
                            <TableBody>
                              {vectorInfo?.chunks.map((chunk) => <TableRow key={chunk.id}>
                                <TableCell className="align-top whitespace-nowrap">#{chunk.id}<br/><span className="text-xs text-slate-500">index {chunk.chunk_index}</span></TableCell>
                                <TableCell className="align-top"><p className="max-w-48 break-words">{chunk.source_label}</p><span className="text-xs text-emerald-300">{chunk.source_type}</span></TableCell>
                                <TableCell className="align-top whitespace-nowrap">{chunk.character_count} chars<br/><span className="text-xs text-slate-500">{chunk.word_count} words</span></TableCell>
                                <TableCell className="min-w-[420px] align-top">
                                  <details><summary className="cursor-pointer text-emerald-300">View full chunk</summary><pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-xl bg-black/30 p-3 text-xs text-slate-200">{chunk.content}</pre></details>
                                  <details className="mt-2"><summary className="cursor-pointer text-xs text-slate-400">View metadata</summary><pre className="mt-2 overflow-auto whitespace-pre-wrap rounded-xl bg-black/30 p-3 text-xs text-slate-300">{JSON.stringify(chunk.metadata, null, 2)}</pre></details>
                                </TableCell>
                              </TableRow>)}
                              {!vectorInfo?.chunks.length && <TableRow><TableCell colSpan={4} className="text-center text-slate-400">No matching chunks.</TableCell></TableRow>}
                            </TableBody>
                          </Table>
                        </div>
                        {vectorInfo && <div className="mt-4 flex items-center justify-between text-sm text-slate-300">
                          <span>{vectorInfo.statistics.filtered_chunks} matching chunks · Page {vectorInfo.pagination.page} of {vectorInfo.pagination.total_pages}</span>
                          <div className="flex gap-2"><Button size="sm" variant="outline" disabled={loadingChunks || chunkPage <= 1} onClick={() => loadChunks(chunkPage - 1)}>Previous</Button><Button size="sm" variant="outline" disabled={loadingChunks || chunkPage >= vectorInfo.pagination.total_pages} onClick={() => loadChunks(chunkPage + 1)}>Next</Button></div>
                        </div>}
                      </div>

                      <div className="rounded-[1.75rem] border border-white/10 bg-white/5 p-6">
                        <div className="flex items-center gap-3 text-emerald-300">
                          <Trash2 />
                          <div>
                            <h2 className="text-lg font-semibold text-white">Vector database contents</h2>
                            <p className="mt-1 text-sm text-slate-400">Documents stored in the knowledge base. Delete any stale source and rebuild if needed.</p>
                          </div>
                        </div>

                        <div className="mt-4 overflow-auto rounded-2xl border border-white/10 bg-slate-950/80">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Filename</TableHead>
                                <TableHead>Type</TableHead>
                                <TableHead>Created</TableHead>
                                <TableHead>Actions</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {documents.map((doc) => (
                                <TableRow key={doc.id}>
                                  <TableCell>{doc.filename}</TableCell>
                                  <TableCell>{doc.source_type}</TableCell>
                                  <TableCell>{new Date(doc.created_at).toLocaleString()}</TableCell>
                                  <TableCell>
                                    <Button variant="outline" size="sm" onClick={() => handleDeleteDocument(doc.id)}>
                                      Delete
                                    </Button>
                                  </TableCell>
                                </TableRow>
                              ))}
                              {!documents.length && (
                                <TableRow>
                                  <TableCell colSpan={4} className="text-center text-sm text-slate-400">No documents in the vector database yet.</TableCell>
                                </TableRow>
                              )}
                            </TableBody>
                          </Table>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default AgentManager;
