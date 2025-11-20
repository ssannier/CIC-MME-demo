"use client";

import React, { useEffect, useRef, useState } from "react";

type Message = {
	id: string;
	role: "user" | "assistant" | "system";
	text: string;
	sources?: Array<{
		key: string;
		similarity: number;
		text_preview: string;
	}>;
};

const ChatWindow: React.FC = () => {
	const [messages, setMessages] = useState<Message[]>([]);
	const [input, setInput] = useState("");
	const [sending, setSending] = useState(false);
	const bottomRef = useRef<HTMLDivElement | null>(null);
	const textareaRef = useRef<HTMLTextAreaElement | null>(null);

	useEffect(() => {
		bottomRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages]);

	const sendMessage = async () => {
		if (!input.trim() || sending) return;
		const text = input.trim();
		const userMsg: Message = { id: String(Date.now()), role: "user", text };
		setMessages((s) => [...s, userMsg]);
		setInput("");
		setSending(true);

		try {
			const apiUrl = (process.env.NEXT_PUBLIC_QUERY_URL as string) || "/api/chat";
			const res = await fetch(apiUrl, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ query: text }),
			});

			if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

			const data = await res.json();
			const assistantText = data?.answer || data?.reply || String(data?.text || "No response");
			const assistantMsg: Message = { 
				id: String(Date.now() + 1), 
				role: "assistant", 
				text: assistantText,
				sources: data?.sources || []
			};
			setMessages((s) => [...s, assistantMsg]);
		} catch (err) {
			setMessages((s) => [...s, { id: String(Date.now() + 2), role: "assistant", text: `Error: ${String(err)}` }]);
		} finally {
			setSending(false);
		}
	};

	const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
	};

	// Auto-resize textarea height (clamps between MIN_HEIGHT and MAX_HEIGHT)
	const MIN_HEIGHT = 48; // px minimum height of textarea
	const MAX_HEIGHT = 200; // px maximum height to grow to

	const adjustTextareaHeight = (el: HTMLTextAreaElement | null) => {
		if (!el) return;
		el.style.height = "auto";
		const needed = el.scrollHeight;
		const clamped = Math.min(Math.max(needed, MIN_HEIGHT), MAX_HEIGHT);
		el.style.height = `${clamped}px`;
		el.style.overflowY = needed > MAX_HEIGHT ? "auto" : "hidden";
	};

	useEffect(() => {
		adjustTextareaHeight(textareaRef.current);
	}, [input]);

	return (
		<div className="w-full max-w-4xl h-[90vh] rounded-xl bg-white dark:bg-zinc-800 shadow-lg flex flex-col overflow-hidden">
			<div className="px-4 py-3 border-b border-gray-100 dark:border-zinc-700">
				<div className="w-full flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
					<h1 className="text-lg font-semibold">Nova Chatbot Demo</h1>
					<div className="text-sm text-gray-500">Connected to: CIC Drive</div>
				</div>
			</div>

			<div className="flex-1 overflow-auto p-6 space-y-4">
				{messages.length === 0 && <div className="text-center text-gray-500">Ask me anything — start the conversation.</div>}

				{messages.map((m) => (
					<div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start flex-col"}>
						<div
							className={
								(m.role === "user" ? "bg-blue-500 text-white rounded-lg rounded-tr-none" : "bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-200 rounded-lg rounded-tl-none") +
								" px-4 py-3 max-w-[70%]"
							}
						>
							<div className="whitespace-pre-wrap">{m.text}</div>
						</div>
						
						{m.sources && m.sources.length > 0 && (
							<div className="mt-2 max-w-[70%] space-y-2">
								<div className="text-xs text-gray-500 dark:text-gray-400 font-medium">Sources:</div>
								{m.sources.map((source, idx) => (
									<div key={idx} className="text-xs bg-gray-50 dark:bg-zinc-900 rounded p-2 border border-gray-200 dark:border-zinc-700">
										<div className="flex justify-between items-center mb-1">
											<span className="font-medium text-gray-700 dark:text-gray-300 truncate">{source.key}</span>
											<span className="text-blue-600 dark:text-blue-400 ml-2 flex-shrink-0">
												{(source.similarity * 100).toFixed(0)}%
											</span>
										</div>
										<p className="text-gray-600 dark:text-gray-400 line-clamp-2">{source.text_preview}</p>
									</div>
								))}
							</div>
						)}
					</div>
				))}

				<div ref={bottomRef} />
			</div>

			<div className="p-4">
				<div className="flex gap-3 items-end">
					<textarea
						ref={textareaRef}
						aria-label="Message"
						value={input}
						onChange={(e) => setInput(e.target.value)}
						onKeyDown={onKeyDown}
						placeholder={sending ? "Waiting for response..." : "Type your message and press Enter"}
						className="flex-1 rounded-md border border-gray-200 dark:border-zinc-700 resize-none px-3 py-2 bg-transparent focus:outline-none"
					/>
          <button
						onClick={sendMessage}
						disabled={sending || !input.trim()}
						className="h-10 px-4 rounded-md bg-black text-white dark:bg-white dark:text-black disabled:opacity-50"
					>
						{sending ? "Sending..." : "Send"}
					</button>
				</div>
			</div>
		</div>
	);
};

export default ChatWindow;
