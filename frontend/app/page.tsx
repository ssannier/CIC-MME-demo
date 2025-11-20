import ChatWindow from "../components/ChatWindow";

export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-zinc-800 p-8 ">
      <div className="w-full flex-1 flex items-center justify-center ">
        <ChatWindow />
      </div>
    </div>
  );
}
